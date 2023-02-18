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
        std::vector<Node*>& nodes = mesh->getNodes();
        handles.resize(handleIndices.size());
        for (int i = 0; i < handleIndices.size(); i++) {
            int index = handleIndices[i];
            nodes[index]->preserve = true;
            handles[i] = new Handle(nodes[index], nodes[index]->x);
        }
    } else {
        thrust::device_vector<int> handleIndicesGpu = handleIndices;
        thrust::device_vector<Node*>& nodes = mesh->getNodesGpu();
        handlesGpu.resize(handleIndicesGpu.size());
        initializeHandles<<<GRID_SIZE, BLOCK_SIZE>>>(handleIndicesGpu.size(), pointer(handleIndicesGpu), pointer(nodes), pointer(handlesGpu));
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
    std::vector<Node*>& nodes = mesh->getNodes();
    int n = nodes.size();

    A.resize(3 * n, 3 * n);
    A.setZero();
    for (int i = 0; i < n; i++) {
        float mass = nodes[i]->mass;
        for (int j = 0; j < 3; j++)
            A.coeffRef(3 * i + j, 3 * i + j) = mass;
    }

    b.resize(3 * n);
    b.setZero();
}

void Cloth::addExternalForces(float dt, const Vector3f& gravity, const Wind* wind, Eigen::SparseMatrix<float>& A, Eigen::VectorXf& b) const {
    std::vector<Node*>& nodes = mesh->getNodes();
    for (int i = 0; i < nodes.size(); i++) {
        Vector3f g = dt * nodes[i]->mass * gravity;
        for (int j = 0; j < 3; j++)
            b(3 * i + j) += g(j);
    }

    std::vector<Face*>& faces = mesh->getFaces();
    Vector3f velocity = wind->getVelocity();
    float density = wind->getDensity();
    for (const Face* face : faces) {
        Vector3f n = face->n;
        float area = face->area;
        Node* node0 = face->vertices[0]->node;
        Node* node1 = face->vertices[1]->node;
        Node* node2 = face->vertices[2]->node;
        Vector3f average = (node0->v + node1->v + node2->v) / 3.0f;
        Vector3f relative = velocity - average;
        float vn = n.dot(relative);
        Vector3f vt = relative - vn * n;
        Vector3f force = area * (density * abs(vn) * vn * n + wind->getDrag() * vt) / 3.0f;
        Vector3f f = dt * force;
        for (int i = 0; i < 3; i++) {
            b(3 * node0->index + i) += f(i);
            b(3 * node1->index + i) += f(i);
            b(3 * node2->index + i) += f(i);
        }
    }
}

void Cloth::addInternalForces(float dt, Eigen::SparseMatrix<float>& A, Eigen::VectorXf& b) const {
    std::vector<Face*>& faces = mesh->getFaces();
    for (const Face* face : faces) {
        Node* node0 = face->vertices[0]->node;
        Node* node1 = face->vertices[1]->node;
        Node* node2 = face->vertices[2]->node;
        Vector9f v(node0->v, node1->v, node2->v);
        
        Vector9f f;
        Matrix9x9f J;
        stretchingForce(face, material, f, J);

        Vector3i indices(node0->index, node1->index, node2->index);
        addSubMatrix(-dt * dt * J, indices, A);
        addSubVector(dt * (f + dt * J * v), indices, b);
    }

    std::vector<Edge*>& edges = mesh->getEdges();
    for (const Edge* edge : edges)
        if (!edge->isBoundary()) {
            Node* node0 = edge->nodes[0];
            Node* node1 = edge->nodes[1];
            Node* node2 = edge->opposites[0]->node;
            Node* node3 = edge->opposites[1]->node;
            Vector12f v(node0->v, node1->v, node2->v, node3->v);

            Vector12f f;
            Matrix12x12f J;
            bendingForce(edge, material, f, J);

            Vector4i indices(node0->index, node1->index, node2->index, node3->index);
            addSubMatrix(-dt * dt * J, indices, A);
            addSubVector(dt * (f + dt * J * v), indices, b);
        }
}

void Cloth::addHandleForces(float dt, float stiffness, Eigen::SparseMatrix<float>& A, Eigen::VectorXf& b) const {
    for (const Handle* handle : handles) {
        Node* node = handle->getNode();
        Vector3f position = handle->getPosition();
        int index = node->index;
        A.coeffRef(3 * index, 3 * index) += dt * dt * stiffness;
        A.coeffRef(3 * index + 1, 3 * index + 1) += dt * dt * stiffness;
        A.coeffRef(3 * index + 2, 3 * index + 2) += dt * dt * stiffness;
        Vector3f f = dt * ((position - node->x) - dt * node->v) * stiffness;
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
        Plane plane = planes[face->vertices[i]->node->index];
        if (plane.n.norm2() == 0.0f)
            continue;
        float h[3];
        for (int j = 0; j < 3; j++)
            h[j] = (face->vertices[j]->node->x - plane.p).dot(plane.n);
        Vector2f dh = face->inverse.transpose() * Vector2f(h[1] - h[0], h[2] - h[0]);
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
    Node* node0 = face->vertices[0]->node;
    Node* node1 = face->vertices[1]->node;
    Node* node2 = face->vertices[2]->node;
    Matrix2x2f M[5];

    Matrix2x2f Sw1 = face->curvature();
    M[0] = (Sw1.transpose() * Sw1) / sqr(remeshing->refineAngle);
    Matrix3x2f Sw2 = face->derivative(node0->n, node1->n, node2->n);
    M[1] = (Sw2.transpose() * Sw2) / sqr(remeshing->refineAngle);
    Matrix3x2f V = face->derivative(node0->v, node1->v, node2->v);
    M[2] = (V.transpose() * V) / sqr(remeshing->refineVelocity);
    Matrix3x2f F = face->derivative(node0->x, node1->x, node2->x);
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
    if (vertex0 == nullptr || vertex1 == nullptr)
        return 0.0f;
    Vector2f du = vertex0->u - vertex1->u;
    return sqrt(0.5f * (du.dot(vertex0->sizing * du) + du.dot(vertex1->sizing * du)));
}

float Cloth::edgeMetric(const Edge* edge) const {
    return max(edgeMetric(edge->vertices[0][0], edge->vertices[0][1]), edgeMetric(edge->vertices[1][0], edge->vertices[1][1]));
}

bool Cloth::shouldFlip(const Edge* edge) const {
    Vertex* vertex0 = edge->vertices[0][0];
    Vertex* vertex1 = edge->vertices[1][1];
    Vertex* vertex2 = edge->opposites[0];
    Vertex* vertex3 = edge->opposites[1];

    Vector2f x = vertex0->u, y = vertex1->u, z = vertex2->u, w = vertex3->u;
    Matrix2x2f M = 0.25f * (vertex0->sizing + vertex1->sizing + vertex2->sizing + vertex3->sizing);
    float area0 = edge->adjacents[0]->area;
    float area1 = edge->adjacents[1]->area;
    return area1 * (x - z).dot(M * (y - z)) + area0 * (y - w).dot(M * (x - w)) < -remeshing->flipThreshold * (area0 + area1);
}

std::vector<Edge*> Cloth::findEdgesToFlip(const std::vector<Edge*>& edges) const {
    std::vector<Edge*> ans;
    for (Edge* edge : edges)
        if (!edge->isBoundary() && !edge->isSeam() && shouldFlip(edge))
            ans.push_back(edge);

    return ans;
}

std::vector<Edge*> Cloth::independentEdges(const std::vector<Edge*>& edges) const {
    std::unordered_set<Node*> nodes;
    std::vector<Edge*> ans;
    for (Edge* edge : edges) {
        Node* node0 = edge->nodes[0];
        Node* node1 = edge->nodes[1];
        if (nodes.find(node0) == nodes.end() && nodes.find(node1) == nodes.end()) {
            ans.push_back(edge);
            nodes.insert(node0);
            nodes.insert(node1);
        }
    }
    return ans;
}

bool Cloth::flipSomeEdges(std::vector<Edge*>& edges, std::vector<Edge*>* edgesToUpdate, std::unordered_map<Node*, std::vector<Edge*>>* adjacentEdges, std::unordered_map<Vertex*, std::vector<Face*>>* adjacentFaces) const {
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

void Cloth::flipEdges(std::vector<Edge*>& edges, std::vector<Edge*>* edgesToUpdate, std::unordered_map<Node*, std::vector<Edge*>>* adjacentEdges, std::unordered_map<Vertex*, std::vector<Face*>>* adjacentFaces) const {
    for (int i = 0; i < 2 * edges.size(); i++)
        if (!flipSomeEdges(edges, edgesToUpdate, adjacentEdges, adjacentFaces))
            return;
}

std::vector<Edge*> Cloth::findEdgesToSplit() const {
    std::vector<Edge*>& edges = mesh->getEdges();
    std::vector<std::pair<float, Edge*>> sorted;
    for (Edge* edge : edges) {
        float m = edgeMetric(edge);
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
            op.split(edge, material, mesh->getNodes().size());
            mesh->apply(op);
            flipEdges(op.activeEdges, &edgesToSplit, nullptr, nullptr);
        }
    return !edgesToSplit.empty();
}

void Cloth::splitEdges() {
    while (splitSomeEdges());
}

bool Cloth::shouldCollapse(std::unordered_map<Node*, std::vector<Edge*>>& adjacentEdges, std::unordered_map<Vertex*, std::vector<Face*>>& adjacentFaces, const Edge* edge, int side) const {
    Node* node = edge->nodes[side];
    if (node->preserve)
        return false;
    
    bool flag = false;
    std::vector<Edge*>& edges = adjacentEdges[node];
    for (const Edge* edge : edges)
        if (edge->isBoundary() || edge->isSeam()) {
            flag = true;
            break;
        }
    if (flag && (!edge->isBoundary() && !edge->isSeam()))
        return false;
    
    if (edge->isSeam())
        for (int i = 0; i < 2; i++) {
            Vertex* vertex0 = edge->vertices[i][side];
            Vertex* vertex1 = edge->vertices[i][1 - side];
            
            std::vector<Face*>& faces = adjacentFaces[vertex0];
            for (const Face* face : faces) {
                Vertex* v0 = face->vertices[0];
                Vertex* v1 = face->vertices[1];
                Vertex* v2 = face->vertices[2];
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
                float aspect = 12.0f * sqrt(3.0f) * a / sqr(p);
                if (a < 1e-6f || aspect < remeshing->aspectMin)
                    return false;
                if (edgeMetric(v0, v1) > 0.9f || edgeMetric(v0, v2) > 0.9f)
                    return false;
            }
        }
    else {
        int index = edge->opposites[0] != nullptr ? 0 : 1;
        Vertex* vertex0 = edge->vertices[index][side];
        Vertex* vertex1 = edge->vertices[index][1 - side];

        std::vector<Face*>& faces = adjacentFaces[vertex0];
        for (const Face* face : faces) {
            Vertex* v0 = face->vertices[0];
            Vertex* v1 = face->vertices[1];
            Vertex* v2 = face->vertices[2];
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
            float aspect = 12.0f * sqrt(3.0f) * a / sqr(p);
            if (a < 1e-6f || aspect < remeshing->aspectMin)
                return false;
            if (edgeMetric(v0, v1) > 0.9f || edgeMetric(v0, v2) > 0.9f)
                return false;
        }
    }

    return true;
}

bool Cloth::collapseSomeEdges(std::unordered_map<Node*, std::vector<Edge*>>& adjacentEdges, std::unordered_map<Vertex*, std::vector<Face*>>& adjacentFaces) const {
    bool flag = false;
    std::vector<Edge*> edges = mesh->getEdges();
    for (const Edge* edge : edges)
        if (edge != nullptr) {
            Operator op;
            if (shouldCollapse(adjacentEdges, adjacentFaces, edge, 0))
                op.collapse(edge, 0, material, adjacentEdges, adjacentFaces);
            else if (shouldCollapse(adjacentEdges, adjacentFaces, edge, 1))
                op.collapse(edge, 1, material, adjacentEdges, adjacentFaces);
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
    std::unordered_map<Node*, std::vector<Edge*>> adjacentEdges;
    for (Edge* edge : edges)
        for (int i = 0; i < 2; i++)
            adjacentEdges[edge->nodes[i]].push_back(edge);
            
    std::vector<Face*>& faces = mesh->getFaces();
    std::unordered_map<Vertex*, std::vector<Face*>> adjacentFaces;
    for (Face* face : faces)
        for (int i = 0; i < 3; i++)
            adjacentFaces[face->vertices[i]].push_back(face);
    
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

        std::vector<Node*>& nodes = mesh->getNodes();
        for (int i = 0; i < nodes.size(); i++) {
            Node* node = nodes[i];
            node->x0 = node->x;
            node->v += Vector3f(dv(3 * i), dv(3 * i + 1), dv(3 * i + 2));
            node->x += node->v * dt;
        }
    } else {
        thrust::device_vector<Node*>& nodes = mesh->getNodesGpu();
        thrust::device_vector<Edge*>& edges = mesh->getEdgesGpu();
        thrust::device_vector<Face*>& faces = mesh->getFacesGpu();

        int nNodes = nodes.size();
        Node** nodesPointer = pointer(nodes);
        int nEdges = edges.size();
        Edge** edgesPointer = pointer(edges);
        int nFaces = faces.size();
        Face** facesPointer = pointer(faces);
        int nHandles = handlesGpu.size();

        int aSize = 3 * nNodes + 144 * nEdges + 81 * nFaces + 3 * nHandles;
        int bSize = 3 * nNodes + 12 * nEdges + 18 * nFaces + 3 * nHandles;

        thrust::device_vector<PairIndex> aIndices(aSize);
        thrust::device_vector<int> bIndices(bSize);
        thrust::device_vector<float> aValues(aSize), bValues(bSize);

        addMass<<<GRID_SIZE, BLOCK_SIZE>>>(nNodes, nodesPointer, pointer(aIndices), pointer(aValues));
        CUDA_CHECK_LAST();

        addGravity<<<GRID_SIZE, BLOCK_SIZE>>>(nNodes, nodesPointer, dt, gravity, pointer(bIndices), pointer(bValues));
        CUDA_CHECK_LAST();

        addWindForces<<<GRID_SIZE, BLOCK_SIZE>>>(nFaces, facesPointer, dt, wind, pointer(bIndices, 3 * nNodes), pointer(bValues, 3 * nNodes));
        CUDA_CHECK_LAST();

        addStretchingForces<<<GRID_SIZE, BLOCK_SIZE>>>(nFaces, facesPointer, dt, material, pointer(aIndices, 3 * nNodes), pointer(aValues, 3 * nNodes), pointer(bIndices, 3 * nNodes + 9 * nFaces), pointer(bValues, 3 * nNodes + 9 * nFaces));
        CUDA_CHECK_LAST();

        addBendingForces<<<GRID_SIZE, BLOCK_SIZE>>>(nEdges, edgesPointer, dt, material, pointer(aIndices, 3 * nNodes + 81 * nFaces), pointer(aValues, 3 * nNodes + 81 * nFaces), pointer(bIndices, 3 * nNodes + 18 * nFaces), pointer(bValues, 3 * nNodes + 18 * nFaces));
        CUDA_CHECK_LAST();

        addHandleForcesGpu<<<GRID_SIZE, BLOCK_SIZE>>>(nHandles, pointer(handlesGpu), dt, handleStiffness, pointer(aIndices, 3 * nNodes + 144 * nEdges + 81 * nFaces), pointer(aValues, 3 * nNodes + 144 * nEdges + 81 * nFaces), pointer(bIndices, 3 * nNodes + 12 * nEdges + 18 * nFaces), pointer(bValues, 3 * nNodes + 12 * nEdges + 18 * nFaces));
        CUDA_CHECK_LAST();

        int n = 3 * nNodes;

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
        setVector<<<GRID_SIZE, BLOCK_SIZE>>>(outputBSize, pointer(outputBIndices), pointer(outputBValues), pointer(b));
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

        updateNodes<<<GRID_SIZE, BLOCK_SIZE>>>(nNodes, dt, pointer(dv), nodesPointer);
        CUDA_CHECK_LAST();
    }
}

void Cloth::remeshingStep(const std::vector<BVH*>& obstacleBvhs, float thickness) {
    std::vector<Node*>& nodes = mesh->getNodes();
    std::vector<Plane> planes(nodes.size(), Plane(Vector3f(0.0f, 0.0f, 0.0f), Vector3f(0.0f, 0.0f, 0.0f)));
    for (int i = 0; i < nodes.size(); i++) {
        Node* node = nodes[i];
        NearPoint point(thickness, node->x);
        for (const BVH* obstacleBvh : obstacleBvhs)
            obstacleBvh->findNearestPoint(node->x, point);
    
        if ((point.x - node->x).norm2() > 1e-8f) {
            planes[i].p = point.x;
            planes[i].n = (node->x - point.x).normalized();
        }
    }

    std::vector<Vertex*>& vertices = mesh->getVertices();
    for (Vertex* vertex : vertices) {
        vertex->area = 0.0f;
        vertex->sizing = Matrix2x2f();
    }

    std::vector<Face*>& faces = mesh->getFaces();
    for (Face* face : faces) {
        float area = face->area;
        Matrix2x2f sizing = faceSizing(face, planes);
        for (int i = 0; i < 3; i++) {
            Vertex* vertex = face->vertices[i];
            vertex->area += face->area;
            vertex->sizing += area * sizing;
        }
    }

    for (Vertex* vertex : vertices)
        vertex->sizing /= vertex->area;

    std::vector<Edge*> edges = mesh->getEdges();
    flipEdges(edges, nullptr, nullptr, nullptr);
    splitEdges();
    collapseEdges();
}

void Cloth::updateStructures() {
    mesh->updateStructures();
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
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    edgeShader->use();
    edgeShader->setMat4("model", model);
    edgeShader->setMat4("view", view);
    edgeShader->setMat4("projection", projection);
    mesh->render();

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    faceShader->use();
    faceShader->setMat4("model", model);
    faceShader->setMat4("view", view);
    faceShader->setMat4("projection", projection);
    faceShader->setVec3("color", Vector3f(0.6f, 0.7f, 1.0f));
    faceShader->setVec3("cameraPosition", cameraPosition);
    faceShader->setVec3("lightDirection", lightDirection);
    faceShader->setInt("selectedFace", selectedFace);
    mesh->render();
}

void Cloth::printDebugInfo(int selectedFace) {
    mesh->printDebugInfo(selectedFace);
}