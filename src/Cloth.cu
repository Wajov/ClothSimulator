#include "Cloth.cuh"

Cloth::Cloth(const Json::Value& json) {
    Transform* transform = new Transform(json["transform"]);
    Material* materialTemp = new Material(json["materials"]);
    if (!gpu)
        material = materialTemp;
    else {
        CUDA_CHECK(cudaMalloc(&material, sizeof(Material)));
        CUDA_CHECK(cudaMemcpy(material, materialTemp, sizeof(Material), cudaMemcpyHostToDevice));
        delete materialTemp;
    }
    mesh = new Mesh(json["mesh"], transform, material);

    std::vector<int> handleIndices;
    for (const Json::Value& handleJson : json["handles"])
        for (const Json::Value& nodeJson : handleJson["nodes"])
            handleIndices.push_back(parseInt(nodeJson));
    
    remeshing = new Remeshing(json["remeshing"]);

    edgeShader = new Shader("shader/Vertex.glsl", "shader/EdgeFragment.glsl");
    faceShader = new Shader("shader/Vertex.glsl", "shader/FaceFragment.glsl");
    delete transform;

    if (!gpu) {
        std::vector<Vertex*>& vertices = mesh->getVertices();
        handles.resize(handleIndices.size());
        for (int i = 0; i < handleIndices.size(); i++) {
            int index = handleIndices[i];
            vertices[index]->preserve = true;
            handles[i] = new Handle(vertices[index], vertices[index]->x);
        }
    } else {
        thrust::device_vector<int> handleIndicesGpu = handleIndices;
        thrust::device_vector<Vertex*> vertices = mesh->getVerticesGpu();
        handlesGpu.resize(handleIndicesGpu.size());
        initializeHandles<<<GRID_SIZE, BLOCK_SIZE>>>(handleIndicesGpu.size(), pointer(handleIndicesGpu), pointer(vertices), pointer(handlesGpu));
        CUDA_CHECK_LAST();

        CUSPARSE_CHECK(cusparseCreate(&cusparseHandle));
        CUSOLVER_CHECK(cusolverSpCreate(&cusolverHandle));
    }
}

Cloth::~Cloth() {
    delete mesh;
    for (const Handle* handle : handles)
        delete handle;
    delete edgeShader;
    delete faceShader;
    
    if (!gpu)
        delete material;
    else {
        CUDA_CHECK(cudaFree(material));
        deleteHandles<<<GRID_SIZE, BLOCK_SIZE>>>(handlesGpu.size(), pointer(handlesGpu));
        CUDA_CHECK_LAST();

        CUSPARSE_CHECK(cusparseDestroy(cusparseHandle));
        CUSOLVER_CHECK(cusolverSpDestroy(cusolverHandle));
    }
}

void Cloth::addSubMatrix(const Matrix9x9f& B, const Vector3i& indices, Eigen::SparseMatrix<float>& A) const {
    for (int i = 0; i < 3; i++) {
        int x = indices(i);
        for (int j = 0; j < 3; j++) {
            int y = indices(j);
            for (int k = 0; k < 3; k++)
                for (int h = 0; h < 3; h++)
                    A.coeffRef(3 * x + k, 3 * y + h) += B(3 * i + k, 3 * j + h);
        }
    }
}

void Cloth::addSubMatrix(const Matrix12x12f& B, const Vector4i& indices, Eigen::SparseMatrix<float>& A) const {
    for (int i = 0; i < 4; i++) {
        int x = indices(i);
        for (int j = 0; j < 4; j++) {
            int y = indices(j);
            for (int k = 0; k < 3; k++)
                for (int h = 0; h < 3; h++)
                    A.coeffRef(3 * x + k, 3 * y + h) += B(3 * i + k, 3 * j + h);
        }
    }
}

void Cloth::addSubVector(const Vector9f& b, const Vector3i& indices, Eigen::VectorXf& a) const {
    for (int i = 0; i < 3; i++) {
        int x = indices(i);
        for (int j = 0; j < 3; j++)
            a(3 * x + j) += b(3 * i + j);
    }
}

void Cloth::addSubVector(const Vector12f& b, const Vector4i& indices, Eigen::VectorXf& a) const {
    for (int i = 0; i < 4; i++) {
        int x = indices(i);
        for (int j = 0; j < 3; j++)
            a(3 * x + j) += b(3 * i + j);
    }
}

void Cloth::init(Eigen::SparseMatrix<float>& A, Eigen::VectorXf& b) const {
    std::vector<Vertex*>& vertices = mesh->getVertices();
    int n = vertices.size();

    A.resize(3 * n, 3 * n);
    A.setZero();
    for (int i = 0; i < n; i++) {
        float m = vertices[i]->m;
        for (int j = 0; j < 3; j++)
            A.coeffRef(3 * i + j, 3 * i + j) = m;
    }

    b.resize(3 * n);
    b.setZero();
}

void Cloth::addExternalForces(float dt, const Vector3f& gravity, const Wind* wind, Eigen::SparseMatrix<float>& A, Eigen::VectorXf& b) const {
    std::vector<Vertex*>& vertices = mesh->getVertices();
    for (int i = 0; i < vertices.size(); i++) {
        Vector3f g = dt * vertices[i]->m * gravity;
        for (int j = 0; j < 3; j++)
            b(3 * i + j) += g(j);
    }

    std::vector<Face*>& faces = mesh->getFaces();
    Vector3f velocity = wind->getVelocity();
    float density = wind->getDensity();
    for (const Face* face : faces) {
        float area = face->getArea();
        Vector3f normal = face->getNormal();
        Vector3f average = (face->getVertex(0)->v + face->getVertex(1)->v + face->getVertex(2)->v) / 3.0f;
        Vector3f relative = velocity - average;
        float vn = normal.dot(relative);
        Vector3f vt = relative - vn * normal;
        Vector3f force = area * (density * abs(vn) * vn * normal + wind->getDrag() * vt) / 3.0f;
        Vector3f f = dt * force;
        for (int i = 0; i < 3; i++) {
            b(3 * face->getVertex(0)->index + i) += f(i);
            b(3 * face->getVertex(1)->index + i) += f(i);
            b(3 * face->getVertex(2)->index + i) += f(i);
        }
    }
}

void Cloth::addInternalForces(float dt, Eigen::SparseMatrix<float>& A, Eigen::VectorXf& b) const {
    std::vector<Face*>& faces = mesh->getFaces();
    for (const Face* face : faces) {
        Vertex* vertex0 = face->getVertex(0);
        Vertex* vertex1 = face->getVertex(1);
        Vertex* vertex2 = face->getVertex(2);
        Vector9f v(vertex0->v, vertex1->v, vertex2->v);
        
        Vector9f f;
        Matrix9x9f J;
        stretchingForce(face, material, f, J);

        Vector3i indices(vertex0->index, vertex1->index, vertex2->index);
        addSubMatrix(-dt * dt * J, indices, A);
        addSubVector(dt * (f + dt * J * v), indices, b);
    }

    std::vector<Edge*>& edges = mesh->getEdges();
    for (const Edge* edge : edges)
        if (!edge->isBoundary()) {
            Vertex* vertex0 = edge->getVertex(0);
            Vertex* vertex1 = edge->getVertex(1);
            Vertex* vertex2 = edge->getOpposite(0);
            Vertex* vertex3 = edge->getOpposite(1);
            Vector12f v(vertex0->v, vertex1->v, vertex2->v, vertex3->v);

            Vector12f f;
            Matrix12x12f J;
            bendingForce(edge, material, f, J);

            Vector4i indices(vertex0->index, vertex1->index, vertex2->index, vertex3->index);
            addSubMatrix(-dt * dt * J, indices, A);
            addSubVector(dt * (f + dt * J * v), indices, b);
        }
}

void Cloth::addHandleForces(float dt, float stiffness, Eigen::SparseMatrix<float>& A, Eigen::VectorXf& b) const {
    for (const Handle* handle : handles) {
        Vertex* vertex = handle->getVertex();
        Vector3f position = handle->getPosition();
        int index = vertex->index;
        A.coeffRef(3 * index, 3 * index) += dt * dt * stiffness;
        A.coeffRef(3 * index + 1, 3 * index + 1) += dt * dt * stiffness;
        A.coeffRef(3 * index + 2, 3 * index + 2) += dt * dt * stiffness;
        Vector3f f = dt * ((position - vertex->x) - dt * vertex->v) * stiffness;
        for (int i = 0; i < 3; i++)
            b(3 * index + i, 0) += f(i);
    }
    // for (const Constraint* constraint : constraints) {
    //     std::vector<Gradient*> gradient = constraint->energyGradient();
    //     std::vector<Hessian*> hessian = constraint->energyHessian();
    //     for (const Gradient* grad : gradient)
    //         b.block<3, 1>(3 * grad->getIndex(), 0) -= dt * grad->getValue();
    //     for (const Hessian* hess : hessian) {
    //         A.block<
    //     }
    // }
}

Matrix2x2f Cloth::compressionMetric(const Matrix2x2f& G, const Matrix2x2f& S2) const {
    Matrix2x2f P(Vector2f(S2(1, 1), -S2(1, 0)), Vector2f(-S2(0, 1), S2(0, 0)));
    Matrix2x2f D = G.transpose() * G - 4.0f * sqr(remeshing->refineCompression) * P * remeshing->ribStiffening;
    return max(-G + sqrt(D), 0.0f) / (2.0f * sqr(remeshing->refineCompression));
}

Matrix2x2f Cloth::obstacleMetric(const Face* face, const std::vector<Plane>& planes) const {
    Matrix2x2f ans;
    for (int i = 0; i < 3; i++) {
        Plane plane = planes[face->getVertex(i)->index];
        if (plane.n.norm2() == 0.0f)
            continue;
        float h[3];
        for (int j = 0; j < 3; j++)
            h[j] = (face->getVertex(j)->x - plane.p).dot(plane.n);
        Vector2f dh = face->getInverse().transpose() * Vector2f(h[1] - h[0], h[2] - h[0]);
        ans += dh.outer(dh) / sqr(h[i]);
    }
    return ans / 3.0f;
}

Matrix2x2f Cloth::maxTensor(const Matrix2x2f M[]) const {
    int n = 0;
    Disk d[5];
    for (int i = 0; i < 5; i++)
        if (M[i].trace() != 0.0f) {
            d[n].o = Vector2f(0.5f * (M[i](0, 0) - M[i](1, 1)), 0.5f * (M[i](0, 1) + M[i](1, 0)));
            d[n].r = 0.5f * (M[i](0, 0) + M[i](1, 1));
            n++;
        }

    Disk disk;
    disk = d[0];
    for (int i = 1; i < n; i++)
        if (!disk.enclose(d[i])) {
            disk = d[i];
            for (int j = 0; j < i; j++)
                if (!disk.enclose(d[j])) {
                    disk = Disk::circumscribedDisk(d[i], d[j]);
                    for (int k = 0; k < j; k++)
                        if (!disk.enclose(d[k]))
                            disk = Disk::circumscribedDisk(d[i], d[j], d[k]);
                }
        }

    Matrix2x2f ans;
    ans(0, 0) = disk.r + disk.o(0);
    ans(0, 1) = ans(1, 0) = disk.o(1);
    ans(1, 1) = disk.r - disk.o(0);
    return ans;
}

Matrix2x2f Cloth::faceSizing(const Face* face, const std::vector<Plane>& planes) const {
    Vertex* vertex0 = face->getVertex(0);
    Vertex* vertex1 = face->getVertex(1);
    Vertex* vertex2 = face->getVertex(2);
    Matrix2x2f M[5];

    Matrix2x2f Sw1 = face->curvature();
    M[0] = (Sw1.transpose() * Sw1) / sqr(remeshing->refineAngle);
    Matrix3x2f Sw2 = face->derivative(vertex0->n, vertex1->n, vertex2->n);
    M[1] = (Sw2.transpose() * Sw2) / sqr(remeshing->refineAngle);
    Matrix3x2f V = face->derivative(vertex0->v, vertex1->v, vertex2->v);
    M[2] = (V.transpose() * V) / sqr(remeshing->refineVelocity);
    Matrix3x2f F = face->derivative(vertex0->x, vertex1->x, vertex2->x);
    M[3] = compressionMetric(F.transpose() * F - Matrix2x2f(1.0f), Sw2.transpose() * Sw2);
    M[4] = obstacleMetric(face, planes);
    Matrix2x2f S = maxTensor(M);

    Matrix2x2f Q;
    Vector2f l;
    eigenvalueDecomposition(S, Q, l);
    for (int i = 0; i < 2; i++)
        l(i) = clamp(l(i), 1.0f / sqr(remeshing->sizeMax), 1.0f / sqr(remeshing->sizeMin));
    float lMax = max(l(0), l(1));
    float lMin = lMax * sqr(remeshing->aspectMin);
    for (int i = 0; i < 2; i++)
        l(i) = max(l(i), lMin);
    return Q * diagonal(l) * Q.transpose();
}

float Cloth::edgeMetric(const Vertex* vertex0, const Vertex* vertex1) const {
    Vector2f du = vertex0->u - vertex1->u;
    return sqrt(0.5f * (du.dot(vertex0->sizing * du) + du.dot(vertex1->sizing * du)));
}

bool Cloth::shouldFlip(const Edge* edge) const {
    Vertex* vertex0 = edge->getVertex(0);
    Vertex* vertex1 = edge->getVertex(1);
    Vertex* vertex2 = edge->getOpposite(0);
    Vertex* vertex3 = edge->getOpposite(1);

    Vector2f x = vertex0->u, y = vertex1->u, z = vertex2->u, w = vertex3->u;
    Matrix2x2f M = 0.25f * (vertex0->sizing + vertex1->sizing + vertex2->sizing + vertex3->sizing);
    float area0 = edge->getAdjacent(0)->getArea();
    float area1 = edge->getAdjacent(1)->getArea();
    return area1 * (x - z).dot(M * (y - z)) + area0 * (y - w).dot(M * (x - w)) < -remeshing->flipThreshold * (area0 + area1);
}

std::vector<Edge*> Cloth::findEdgesToFlip(const std::vector<Edge*>& edges) const {
    std::vector<Edge*> ans;
    for (Edge* edge : edges)
        if (!edge->isBoundary() && shouldFlip(edge))
            ans.push_back(edge);

    return ans;
}

std::vector<Edge*> Cloth::independentEdges(const std::vector<Edge*>& edges) const {
    std::unordered_set<Vertex*> vertices;
    std::vector<Edge*> ans;
    for (Edge* edge : edges) {
        Vertex* vertex0 = edge->getVertex(0);
        Vertex* vertex1 = edge->getVertex(1);
        if (vertices.find(vertex0) == vertices.end() && vertices.find(vertex1) == vertices.end()) {
            ans.push_back(edge);
            vertices.insert(vertex0);
            vertices.insert(vertex1);
        }
    }
    return ans;
}

bool Cloth::flipSomeEdges(std::vector<Edge*>& edges, std::vector<Edge*>* edgesToUpdate, std::unordered_map<Vertex*, std::vector<Edge*>>* adjacentEdges, std::unordered_map<Vertex*, std::vector<Face*>>* adjacentFaces) const {
    static int nEdges = 0;
    std::vector<Edge*> edgesToFlip = std::move(independentEdges(std::move(findEdgesToFlip(edges))));
    if (edgesToFlip.size() == nEdges)
        return false;
    
    nEdges = edgesToFlip.size();
    for (const Edge* edge : edgesToFlip) {
        Operator op;
        op.flip(edge, material);
        op.update(edges);
        if (edgesToUpdate != nullptr)
            op.setNull(*edgesToUpdate);
        if (adjacentEdges != nullptr && adjacentFaces != nullptr)
            op.updateAdjacents(*adjacentEdges, *adjacentFaces);
        mesh->apply(op);
    }
    return !edgesToFlip.empty();
}

void Cloth::flipEdges(std::vector<Edge*>& edges, std::vector<Edge*>* edgesToUpdate, std::unordered_map<Vertex*, std::vector<Edge*>>* adjacentEdges, std::unordered_map<Vertex*, std::vector<Face*>>* adjacentFaces) const {
    for (int i = 0; i < 2 * edges.size(); i++)
        if (!flipSomeEdges(edges, edgesToUpdate, adjacentEdges, adjacentFaces))
            return;
}

std::vector<Edge*> Cloth::findEdgesToSplit() const {
    std::vector<Edge*>& edges = mesh->getEdges();
    std::vector<std::pair<float, Edge*>> sorted;
    for (Edge* edge : edges) {
        float m = edgeMetric(edge->getVertex(0), edge->getVertex(1));
        if (m > 1.0f)
            sorted.push_back(std::make_pair(m, edge));
    }
    std::sort(sorted.begin(), sorted.end(), [](const std::pair<float, Edge*>& a, const std::pair<float, Edge*>& b) {
        return a.first > b.first;
    });

    std::vector<Edge*> ans(sorted.size());
    for (int i = 0; i < sorted.size(); i++)
        ans[i] = sorted[i].second;

    return ans;
}

bool Cloth::splitSomeEdges() const {
    std::vector<Edge*> edgesToSplit = std::move(findEdgesToSplit());
    for (const Edge* edge : edgesToSplit)
        if (edge != nullptr) {
            Operator op;
            op.split(edge, material, mesh->getVertices().size());
            mesh->apply(op);
            flipEdges(op.activeEdges, &edgesToSplit, nullptr, nullptr);
        }
    return !edgesToSplit.empty();
}

void Cloth::splitEdges() {
    while (splitSomeEdges());
}

bool Cloth::shouldCollapse(std::unordered_map<Vertex*, std::vector<Edge*>>& adjacentEdges, std::unordered_map<Vertex*, std::vector<Face*>>& adjacentFaces, const Edge* edge, bool reverse) const {
    Vertex* vertex0 = edge->getVertex(0);
    Vertex* vertex1 = edge->getVertex(1);
    if (reverse)
        mySwap(vertex0, vertex1);

    if (vertex0->preserve)
        return false;
    
    bool flag = false;
    std::vector<Edge*>& edges = adjacentEdges[vertex0];
    for (const Edge* edge : edges)
        if (edge->isBoundary()) {
            flag = true;
            break;
        }
    if (flag && !edge->isBoundary())
        return false;
    
    std::vector<Face*>& faces = adjacentFaces[vertex0];
    for (const Face* face : faces) {
        Vertex* v0 = face->getVertex(0);
        Vertex* v1 = face->getVertex(1);
        Vertex* v2 = face->getVertex(2);
        if (v0 == vertex1 || v1 == vertex1 || v2 == vertex1)
            continue;
        
        if (v0 == vertex0)
            v0 = vertex1;
        else if (v1 == vertex0) {
            v1 = vertex1;
            mySwap(v0, v1);
        } else {
            v2 = vertex1;
            mySwap(v0, v2);
        }
        Vector2f u0 = v0->u;
        Vector2f u1 = v1->u;
        Vector2f u2 = v2->u;
        float a = 0.5f * (u1 - u0).cross(u2 - u0);
        float p = (u0 - u1).norm() + (u1 - u2).norm() + (u2 - u0).norm();
        float aspect = 12.0f * sqrt(3.0f) / sqr(p);
        if (a < 1e-6f || aspect < remeshing->aspectMin)
            return false;
        if (edgeMetric(v0, v1) > 0.9f || edgeMetric(v0, v2) > 0.9f)
            return false;
    }
    return true;
}

bool Cloth::collapseSomeEdges(std::unordered_map<Vertex*, std::vector<Edge*>>& adjacentEdges, std::unordered_map<Vertex*, std::vector<Face*>>& adjacentFaces) const {
    bool flag = false;
    std::vector<Edge*> edges = mesh->getEdges();
    for (const Edge* edge : edges)
        if (edge != nullptr) {
            Operator op;
            if (shouldCollapse(adjacentEdges, adjacentFaces, edge, false))
                op.collapse(edge, false, material, adjacentEdges, adjacentFaces);
            else if (shouldCollapse(adjacentEdges, adjacentFaces, edge, true))
                op.collapse(edge, true, material, adjacentEdges, adjacentFaces);
            else
                continue;
            op.setNull(edges);
            op.updateAdjacents(adjacentEdges, adjacentFaces);
            mesh->apply(op);
            flipEdges(op.activeEdges, &edges, &adjacentEdges, &adjacentFaces);
            flag = true;
        }
    return flag;
}

void Cloth::collapseEdges() const {
    std::vector<Edge*>& edges = mesh->getEdges();
    std::unordered_map<Vertex*, std::vector<Edge*>> adjacentEdges;
    for (Edge* edge : edges)
        for (int i = 0; i < 2; i++)
            adjacentEdges[edge->getVertex(i)].push_back(edge);
            
    std::vector<Face*>& faces = mesh->getFaces();
    std::unordered_map<Vertex*, std::vector<Face*>> adjacentFaces;
    for (Face* face : faces)
        for (int i = 0; i < 3; i++)
            adjacentFaces[face->getVertex(i)].push_back(face);
    
    while (collapseSomeEdges(adjacentEdges, adjacentFaces));
}

Mesh* Cloth::getMesh() const {
    return mesh;
}

void Cloth::readDataFromFile(const std::string& path) {
    mesh->readDataFromFile(path);
    mesh->updateGeometries();
}

void Cloth::physicsStep(float dt, float handleStiffness, const Vector3f& gravity, const Wind* wind) {
    if (!gpu) {
        Eigen::SparseMatrix<float> A;
        Eigen::VectorXf b;

        init(A, b);
        addExternalForces(dt, gravity, wind, A, b);
        addInternalForces(dt, A, b);
        addHandleForces(dt, handleStiffness, A, b);

        Eigen::SimplicialLLT<Eigen::SparseMatrix<float>> cholesky;
        cholesky.compute(A);
        Eigen::VectorXf dv = cholesky.solve(b);

        std::vector<Vertex*>& vertices = mesh->getVertices();
        for (int i = 0; i < vertices.size(); i++) {
            vertices[i]->x0 = vertices[i]->x;
            vertices[i]->v += Vector3f(dv(3 * i), dv(3 * i + 1), dv(3 * i + 2));
            vertices[i]->x += vertices[i]->v * dt;
        }
    } else {
        thrust::device_vector<Vertex*>& vertices = mesh->getVerticesGpu();
        thrust::device_vector<Edge*>& edges = mesh->getEdgesGpu();
        thrust::device_vector<Face*>& faces = mesh->getFacesGpu();

        int nVertices = vertices.size();
        Vertex** verticesPointer = pointer(vertices);
        int nEdges = edges.size();
        Edge** edgesPointer = pointer(edges);
        int nFaces = faces.size();
        Face** facesPointer = pointer(faces);
        int nHandles = handlesGpu.size();

        int aSize = 3 * nVertices + 144 * nEdges + 81 * nFaces + 3 * nHandles;
        int bSize = 3 * nVertices + 12 * nEdges + 18 * nFaces + 3 * nHandles;

        thrust::device_vector<PairIndex> aIndices(aSize);
        thrust::device_vector<int> bIndices(bSize);
        thrust::device_vector<float> aValues(aSize), bValues(bSize);

        addMass<<<GRID_SIZE, BLOCK_SIZE>>>(nVertices, verticesPointer, pointer(aIndices), pointer(aValues));
        CUDA_CHECK_LAST();

        addGravity<<<GRID_SIZE, BLOCK_SIZE>>>(nVertices, verticesPointer, dt, gravity, pointer(bIndices), pointer(bValues));
        CUDA_CHECK_LAST();

        addWindForces<<<GRID_SIZE, BLOCK_SIZE>>>(nFaces, facesPointer, dt, wind, pointer(bIndices, 3 * nVertices), pointer(bValues, 3 * nVertices));
        CUDA_CHECK_LAST();

        addStretchingForces<<<GRID_SIZE, BLOCK_SIZE>>>(nFaces, facesPointer, dt, material, pointer(aIndices, 3 * nVertices), pointer(aValues, 3 * nVertices), pointer(bIndices, 3 * nVertices + 9 * nFaces), pointer(bValues, 3 * nVertices + 9 * nFaces));
        CUDA_CHECK_LAST();

        addBendingForces<<<GRID_SIZE, BLOCK_SIZE>>>(nEdges, edgesPointer, dt, material, pointer(aIndices, 3 * nVertices + 81 * nFaces), pointer(aValues, 3 * nVertices + 81 * nFaces), pointer(bIndices, 3 * nVertices + 18 * nFaces), pointer(bValues, 3 * nVertices + 18 * nFaces));
        CUDA_CHECK_LAST();

        addHandleForcesGpu<<<GRID_SIZE, BLOCK_SIZE>>>(nHandles, pointer(handlesGpu), dt, handleStiffness, pointer(aIndices, 3 * nVertices + 144 * nEdges + 81 * nFaces), pointer(aValues, 3 * nVertices + 144 * nEdges + 81 * nFaces), pointer(bIndices, 3 * nVertices + 12 * nEdges + 18 * nFaces), pointer(bValues, 3 * nVertices + 12 * nEdges + 18 * nFaces));
        CUDA_CHECK_LAST();

        int n = 3 * nVertices;

        thrust::sort_by_key(aIndices.begin(), aIndices.end(), aValues.begin());
        thrust::device_vector<PairIndex> outputAIndices(aSize);
        thrust::device_vector<float> values(aSize);
        auto iter = thrust::reduce_by_key(aIndices.begin(), aIndices.end(), aValues.begin(), outputAIndices.begin(), values.begin());
        int nNonZero = iter.first - outputAIndices.begin();
        thrust::device_vector<int> rowIndices(nNonZero), colIndies(nNonZero);
        splitIndices<<<GRID_SIZE, BLOCK_SIZE>>>(nNonZero, pointer(outputAIndices), pointer(rowIndices), pointer(colIndies));
        CUDA_CHECK_LAST();

        thrust::sort_by_key(bIndices.begin(), bIndices.end(), bValues.begin());
        thrust::device_vector<int> outputBIndices(bSize);
        thrust::device_vector<float> outputBValues(bSize);
        auto jter = thrust::reduce_by_key(bIndices.begin(), bIndices.end(), bValues.begin(), outputBIndices.begin(), outputBValues.begin());
        int outputBSize = jter.first - outputBIndices.begin();
        thrust::device_vector<float> b(n);
        setupVector<<<GRID_SIZE, BLOCK_SIZE>>>(outputBSize, pointer(outputBIndices), pointer(outputBValues), pointer(b));
        CUDA_CHECK_LAST();

        thrust::device_vector<int> rowPointer(n + 1);
        CUSPARSE_CHECK(cusparseXcoo2csr(cusparseHandle, pointer(rowIndices), nNonZero, n, pointer(rowPointer), CUSPARSE_INDEX_BASE_ZERO));

        cusparseMatDescr_t descr;
        CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
        thrust::device_vector<float> dv(n);
        int singularity;
        CUSOLVER_CHECK(cusolverSpScsrlsvchol(cusolverHandle, n, nNonZero, descr, pointer(values), pointer(rowPointer), pointer(colIndies), pointer(b), 1e-5f, 0, pointer(dv), &singularity));
        if (singularity != -1)
            std::cerr << "Not SPD! " << std::endl;

        updateVertices<<<GRID_SIZE, BLOCK_SIZE>>>(nVertices, dt, pointer(dv), verticesPointer);
        CUDA_CHECK_LAST();
    }
}

void Cloth::remeshingStep(const std::vector<BVH*>& obstacleBvhs, float thickness) {
    std::vector<Vertex*>& vertices = mesh->getVertices();
    std::vector<Plane> planes(vertices.size(), Plane(Vector3f(0.0f, 0.0f, 0.0f), Vector3f(0.0f, 0.0f, 0.0f)));
    for (int i = 0; i < vertices.size(); i++) {
        NearPoint point(thickness, vertices[i]->x);
        for (const BVH* obstacleBvh : obstacleBvhs)
            obstacleBvh->findNearestPoint(vertices[i]->x, point);
    
        if ((point.x - vertices[i]->x).norm2() > 1e-8f) {
            planes[i].p = point.x;
            planes[i].n = (vertices[i]->x - point.x).normalized();
        }
    }

    for (Vertex* vertex : vertices)
        vertex->sizing = Matrix2x2f();

    std::vector<Face*>& faces = mesh->getFaces();
    for (Face* face : faces) {
        float area = face->getArea();
        Matrix2x2f sizing = faceSizing(face, planes);
        for (int i = 0; i < 3; i++)
            face->getVertex(i)->sizing += area * sizing;
    }

    for (Vertex* vertex : vertices)
        vertex->sizing /= vertex->a;

    std::vector<Edge*> edges = mesh->getEdges();
    flipEdges(edges, nullptr, nullptr, nullptr);
    splitEdges();
    collapseEdges();
}

void Cloth::updateIndices() {
    mesh->updateIndices();
}

void Cloth::updateGeometries() {
    mesh->updateGeometries();
}

void Cloth::updateVelocities(float dt) {
    mesh->updateVelocities(dt);
}

void Cloth::updateRenderingData(bool rebind) {
    mesh->updateRenderingData(rebind);
}

void Cloth::bind() {
    mesh->bind();
}

void Cloth::render(const Matrix4x4f& model, const Matrix4x4f& view, const Matrix4x4f& projection, const Vector3f& cameraPosition, const Vector3f& lightDirection, int selectedFace) const {
    edgeShader->use();
    edgeShader->setMat4("model", model);
    edgeShader->setMat4("view", view);
    edgeShader->setMat4("projection", projection);
    mesh->renderEdges();

    faceShader->use();
    faceShader->setMat4("model", model);
    faceShader->setMat4("view", view);
    faceShader->setMat4("projection", projection);
    faceShader->setVec3("color", Vector3f(0.6f, 0.7f, 1.0f));
    faceShader->setVec3("cameraPosition", cameraPosition);
    faceShader->setVec3("lightDirection", lightDirection);
    faceShader->setInt("selectedFace", selectedFace);
    mesh->renderFaces();
}

void Cloth::printDebugInfo(int selectedFace) {
    mesh->printDebugInfo(selectedFace);
}