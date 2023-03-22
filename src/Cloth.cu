#include "Cloth.cuh"

Cloth::Cloth(const Json::Value& json, MemoryPool* pool) {
    Transformation transformation(json["transform"]);
    Material* materialTemp = new Material(json["materials"]);
    Remeshing* remeshingTemp = new Remeshing(json["remeshing"]);
    if (!gpu) {
        material = materialTemp;
        remeshing = remeshingTemp;
    } else {
        CUDA_CHECK(cudaMalloc(&material, sizeof(Material)));
        CUDA_CHECK(cudaMemcpy(material, materialTemp, sizeof(Material), cudaMemcpyHostToDevice));
        delete materialTemp;

        CUDA_CHECK(cudaMalloc(&remeshing, sizeof(Remeshing)));
        CUDA_CHECK(cudaMemcpy(remeshing, remeshingTemp, sizeof(Remeshing), cudaMemcpyHostToDevice));
        delete remeshingTemp;
    }
    mesh = new Mesh(parseString(json["mesh"]), transformation, material, pool);

    std::vector<int> handleIndices;
    for (const Json::Value& handleJson : json["handles"])
        for (const Json::Value& nodeJson : handleJson["nodes"])
            handleIndices.push_back(parseInt(nodeJson));
    
    if (!gpu) {
        std::vector<Node*>& nodes = mesh->getNodes();
        handles.resize(handleIndices.size());
        for (int i = 0; i < handleIndices.size(); i++) {
            Node* node = nodes[handleIndices[i]];
            Handle& handle = handles[i];
            node->preserve = true;
            handle.node = node;
            handle.position = node->x;
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
    delete edgeShader;
    delete faceShader;
    
    if (!gpu)
        delete material;
    else {
        CUDA_CHECK(cudaFree(material));

        CUSPARSE_CHECK(cusparseDestroy(cusparseHandle));
        CUSOLVER_CHECK(cusolverSpDestroy(cusolverHandle));
    }
}

void Cloth::addMatrixAndVector(const Matrix9x9f& B, const Vector9f& b, const Vector3i& indices, Eigen::SparseMatrix<float>& A, Eigen::VectorXf& a) const {
    for (int i = 0; i < 3; i++) {
        int x = indices(i);
        if (x > -1) {
            for (int j = 0; j < 3; j++) {
                int y = indices(j);
                if (y > -1)
                    for (int k = 0; k < 3; k++)
                        for (int h = 0; h < 3; h++)
                            A.coeffRef(3 * x + k, 3 * y + h) += B(3 * i + k, 3 * j + h);
            }

            for (int j = 0; j < 3; j++)
                a(3 * x + j) += b(3 * i + j);
        }
    }
}

void Cloth::addMatrixAndVector(const Matrix12x12f& B, const Vector12f& b, const Vector4i& indices, Eigen::SparseMatrix<float>& A, Eigen::VectorXf& a) const {
    for (int i = 0; i < 4; i++) {
        int x = indices(i);
        if (x > -1) {
            for (int j = 0; j < 4; j++) {
                int y = indices(j);
                if (y > -1)
                    for (int k = 0; k < 3; k++)
                        for (int h = 0; h < 3; h++)
                            A.coeffRef(3 * x + k, 3 * y + h) += B(3 * i + k, 3 * j + h);
            }

            for (int j = 0; j < 3; j++)
                a(3 * x + j) += b(3 * i + j);
        }
    }
}

void Cloth::initializeForces(Eigen::SparseMatrix<float>& A, Eigen::VectorXf& b) const {
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
        Vector3i indices(node0->index, node1->index, node2->index);
        
        Vector9f f;
        Matrix9x9f J;
        stretchingForce(face, material, f, J);
        addMatrixAndVector(-dt * dt * J, dt * (f + dt * J * v), indices, A, b);
    }

    std::vector<Edge*>& edges = mesh->getEdges();
    for (const Edge* edge : edges)
        if (!edge->isBoundary()) {
            Node* node0 = edge->nodes[0];
            Node* node1 = edge->nodes[1];
            Node* node2 = edge->opposites[0]->node;
            Node* node3 = edge->opposites[1]->node;
            Vector12f v(node0->v, node1->v, node2->v, node3->v);
            Vector4i indices(node0->index, node1->index, node2->index, node3->index);

            Vector12f f;
            Matrix12x12f J;
            bendingForce(edge, material, f, J);
            addMatrixAndVector(-dt * dt * J, dt * (f + dt * J * v), indices, A, b);
        }
}

void Cloth::addHandleForces(float dt, float stiffness, Eigen::SparseMatrix<float>& A, Eigen::VectorXf& b) const {
    for (const Handle& handle : handles) {
        Node* node = handle.node;
        int index = node->index;
        A.coeffRef(3 * index, 3 * index) += dt * dt * stiffness;
        A.coeffRef(3 * index + 1, 3 * index + 1) += dt * dt * stiffness;
        A.coeffRef(3 * index + 2, 3 * index + 2) += dt * dt * stiffness;
        Vector3f f = dt * ((handle.position - node->x) - dt * node->v) * stiffness;
        for (int i = 0; i < 3; i++)
            b(3 * index + i, 0) += f(i);
    }
}

void Cloth::addProximityForces(float dt, const std::vector<Proximity>& proximities, float thickness, Eigen::SparseMatrix<float>& A, Eigen::VectorXf& b) const {
    for (const Proximity& proximity : proximities) {
        Node* const* nodes = proximity.nodes;
        Vector12f v(nodes[0]->v, nodes[1]->v, nodes[2]->v, nodes[3]->v);
        Vector4i indices;
        for (int i = 0; i < 4; i++)
            indices(i) = mesh->contain(nodes[i]) ? nodes[i]->index : -1;

        float d = -thickness;
        for (int i = 0; i < 4; i++)
            d += proximity.w[i] * proximity.n.dot(nodes[i]->x);
        d = max(-d, 0.0f);

        Vector12f f;
        Matrix12x12f J;
        impulseForce(proximity, d, thickness, f, J);
        addMatrixAndVector(-dt * dt * J, dt * (f + dt * J * v), indices, A, b);

        frictionForce(proximity, d, thickness, dt, f, J);
        addMatrixAndVector(-dt * J, dt * f, indices, A, b);
    }
}

std::vector<Plane> Cloth::findNearestPlane(const std::vector<BVH*>& obstacleBvhs, float thickness) const {
    std::vector<Node*>& nodes = mesh->getNodes();
    std::vector<Plane> ans(nodes.size(), Plane(Vector3f(), Vector3f()));
    for (int i = 0; i < nodes.size(); i++) {
        Node* node = nodes[i];
        NearPoint point(thickness, node->x);
        for (const BVH* obstacleBvh : obstacleBvhs)
            obstacleBvh->findNearestPoint(node->x, point);
    
        Vector3f n = node->x - point.x;
        if (n.norm2() > 1e-8f)
            ans[i] = Plane(point.x, n.normalized());
    }

    return ans;
}

thrust::device_vector<Plane> Cloth::findNearestPlaneGpu(const std::vector<BVH*>& obstacleBvhs, float thickness) const {
    thrust::device_vector<Node*>& nodes = mesh->getNodesGpu();
    int nNodes = nodes.size();
    thrust::device_vector<Vector3f> x(nNodes);
    Vector3f* xPointer = pointer(x);
    setX<<<GRID_SIZE, BLOCK_SIZE>>>(nNodes, pointer(nodes), xPointer);
    CUDA_CHECK_LAST();

    thrust::device_vector<NearPoint> points(nNodes, NearPoint(thickness, Vector3f()));
    NearPoint* pointsPointer = pointer(points);
    initializeNearPoints<<<GRID_SIZE, BLOCK_SIZE>>>(nNodes, xPointer, pointsPointer);
    CUDA_CHECK_LAST();

    for (const BVH* obstacleBvh : obstacleBvhs)
        obstacleBvh->findNearestPoint(x, points);
    thrust::device_vector<Plane> ans(nNodes, Plane(Vector3f(), Vector3f()));
    setNearestPlane<<<GRID_SIZE, BLOCK_SIZE>>>(nNodes, xPointer, pointsPointer, pointer(ans));
    CUDA_CHECK_LAST();

    return ans;
}

void Cloth::computeSizing(const std::vector<Plane>& planes) {
    std::vector<Vertex*>& vertices = mesh->getVertices();
    for (Vertex* vertex : vertices) {
        vertex->area = 0.0f;
        vertex->sizing = Matrix2x2f();
    }

    std::vector<Face*>& faces = mesh->getFaces();
    for (Face* face : faces) {
        float area = face->area;
        Matrix2x2f sizing = faceSizing(face, static_cast<const Plane*>(planes.data()), remeshing);
        for (int i = 0; i < 3; i++) {
            Vertex* vertex = face->vertices[i];
            vertex->area += area;
            vertex->sizing += area * sizing;
        }
    }

    for (Vertex* vertex : vertices)
        vertex->sizing /= vertex->area;
}

void Cloth::computeSizing(const thrust::device_vector<Plane>& planes) {
    thrust::device_vector<Vertex*>& vertices = mesh->getVerticesGpu();
    initializeSizing<<<GRID_SIZE, BLOCK_SIZE>>>(vertices.size(), pointer(vertices));
    CUDA_CHECK_LAST();

    thrust::device_vector<Face*>& faces = mesh->getFacesGpu();
    computeSizingGpu<<<GRID_SIZE, BLOCK_SIZE>>>(faces.size(), pointer(faces), pointer(planes), remeshing);
    CUDA_CHECK_LAST();

    finalizeSizing<<<GRID_SIZE, BLOCK_SIZE>>>(vertices.size(), pointer(vertices));
    CUDA_CHECK_LAST();
}

std::vector<Edge*> Cloth::findEdgesToFlip() const {
    std::vector<Edge*>& edges = mesh->getEdges();
    std::unordered_set<Face*> faces;
    std::vector<Edge*> ans;
    for (Edge* edge : edges)
        if (shouldFlip(edge, remeshing)) {
            Face* face0 = edge->adjacents[0];
            Face* face1 = edge->adjacents[1];
            if (faces.find(face0) == faces.end() && faces.find(face1) == faces.end()) {
                ans.push_back(edge);
                faces.insert(face0);
                faces.insert(face1);
            }
        }

    return ans;
}

thrust::device_vector<Edge*> Cloth::findEdgesToFlipGpu() const {
    thrust::device_vector<Edge*>& edges = mesh->getEdgesGpu();
    int nEdges = edges.size();
    thrust::device_vector<Edge*> edgesToFlip(nEdges);
    Edge** edgesToFlipPointer = pointer(edgesToFlip);
    checkEdgesToFlip<<<GRID_SIZE, BLOCK_SIZE>>>(nEdges, pointer(edges), remeshing, edgesToFlipPointer);
    CUDA_CHECK_LAST();

    edgesToFlip.erase(thrust::remove(edgesToFlip.begin(), edgesToFlip.end(), nullptr), edgesToFlip.end());
    int nEdgesToFlip = edgesToFlip.size();
    thrust::device_vector<Edge*> ans(nEdgesToFlip, nullptr);
    Edge** ansPointer = pointer(ans);
    initializeEdgeFaces<<<GRID_SIZE, BLOCK_SIZE>>>(nEdgesToFlip, edgesToFlipPointer);
    CUDA_CHECK_LAST();

    int num, newNum = nEdgesToFlip;
    do {
        num = newNum;
        resetEdgeFaces<<<GRID_SIZE, BLOCK_SIZE>>>(nEdgesToFlip, edgesToFlipPointer);
        CUDA_CHECK_LAST();

        computeEdgeMinIndices<<<GRID_SIZE, BLOCK_SIZE>>>(nEdgesToFlip, edgesToFlipPointer);
        CUDA_CHECK_LAST();

        checkIndependentEdges<<<GRID_SIZE, BLOCK_SIZE>>>(nEdgesToFlip, edgesToFlipPointer, ansPointer);
        CUDA_CHECK_LAST();

        newNum = thrust::count(ans.begin(), ans.end(), nullptr);
    } while (num > newNum);

    ans.erase(thrust::remove(ans.begin(), ans.end(), nullptr), ans.end());
    return ans;
}

bool Cloth::flipSomeEdges(MemoryPool* pool) {
    static int nEdges = 0;
    Operator op;
    
    if (!gpu) {
        std::vector<Edge*> edgesToFlip = std::move(findEdgesToFlip());
        if (edgesToFlip.empty() || edgesToFlip.size() == nEdges)
            return false;
        
        nEdges = edgesToFlip.size();
        for (const Edge* edge : edgesToFlip)
            op.flip(edge, material, pool);
    } else {
        thrust::device_vector<Edge*> edgesToFlip = std::move(findEdgesToFlipGpu());
        if (edgesToFlip.empty() || edgesToFlip.size() == nEdges)
            return false;
        
        nEdges = edgesToFlip.size();
        op.flip(edgesToFlip, material, pool);
    }

    mesh->apply(op);
    return true;
}

void Cloth::flipEdges(MemoryPool* pool) {
    int nEdges = !gpu ? mesh->getEdges().size() : mesh->getEdgesGpu().size();
    for (int i = 0; i < 2 * nEdges; i++)
        if (!flipSomeEdges(pool))
            return;
}

std::vector<Edge*> Cloth::findEdgesToSplit() const {
    std::vector<Edge*>& edges = mesh->getEdges();
    std::vector<PairfE> sorted;
    for (Edge* edge : edges) {
        float m = edgeMetric(edge);
        if (m > 1.0f)
            sorted.emplace_back(m, edge);
    }
    std::sort(sorted.begin(), sorted.end(), [](const PairfE& a, const PairfE& b) {
        return a.first > b.first;
    });

    std::unordered_set<Face*> faces;
    std::vector<Edge*> ans;
    for (const PairfE& p : sorted) {
        Edge* edge = p.second;
        Face* face0 = edge->adjacents[0];
        Face* face1 = edge->adjacents[1];
        if ((face0 == nullptr || faces.find(face0) == faces.end()) && (face1 == nullptr || faces.find(face1) == faces.end())) {
            ans.push_back(edge);
            if (face0 != nullptr)
                faces.insert(face0);
            if (face1 != nullptr)
                faces.insert(face1);
        }
    }

    return ans;
}

thrust::device_vector<Edge*> Cloth::findEdgesToSplitGpu() const {
    thrust::device_vector<Edge*>& edges = mesh->getEdgesGpu();
    int nEdges = edges.size();
    thrust::device_vector<Edge*> edgesToSplit(nEdges);
    Edge** edgesToSplitPointer = pointer(edgesToSplit);
    thrust::device_vector<float> metrics(nEdges);
    checkEdgesToSplit<<<GRID_SIZE, BLOCK_SIZE>>>(nEdges, pointer(edges), edgesToSplitPointer, pointer(metrics));
    CUDA_CHECK_LAST();

    metrics.erase(thrust::remove_if(metrics.begin(), metrics.end(), edgesToSplit.begin(), IsNull()), metrics.end());
    edgesToSplit.erase(thrust::remove(edgesToSplit.begin(), edgesToSplit.end(), nullptr), edgesToSplit.end());
    thrust::sort_by_key(metrics.begin(), metrics.end(), edgesToSplit.begin(), thrust::greater<float>());
    int nEdgesToSplit = edgesToSplit.size();
    thrust::device_vector<Edge*> ans(nEdgesToSplit, nullptr);
    Edge** ansPointer = pointer(ans);
    initializeEdgeFaces<<<GRID_SIZE, BLOCK_SIZE>>>(nEdgesToSplit, edgesToSplitPointer);
    CUDA_CHECK_LAST();

    int num, newNum = nEdgesToSplit;
    do {
        num = newNum;
        resetEdgeFaces<<<GRID_SIZE, BLOCK_SIZE>>>(nEdgesToSplit, edgesToSplitPointer);
        CUDA_CHECK_LAST();

        computeEdgeMinIndices<<<GRID_SIZE, BLOCK_SIZE>>>(nEdgesToSplit, edgesToSplitPointer);
        CUDA_CHECK_LAST();

        checkIndependentEdges<<<GRID_SIZE, BLOCK_SIZE>>>(nEdgesToSplit, edgesToSplitPointer, ansPointer);
        CUDA_CHECK_LAST();

        newNum = thrust::count(ans.begin(), ans.end(), nullptr);
    } while (num > newNum);

    ans.erase(thrust::remove(ans.begin(), ans.end(), nullptr), ans.end());
    return ans;
}

bool Cloth::splitSomeEdges(MemoryPool* pool) {
    Operator op;

    if (!gpu) {
        std::vector<Edge*> edgesToSplit = std::move(findEdgesToSplit());
        if (edgesToSplit.empty())
            return false;
        
        for (const Edge* edge : edgesToSplit)
            op.split(edge, material, pool);
    } else {
        thrust::device_vector<Edge*> edgesToSplit = std::move(findEdgesToSplitGpu());
        if (edgesToSplit.empty())
            return false;

        op.split(edgesToSplit, material, pool);
    }

    mesh->apply(op);
    flipEdges(pool);
    return true;
}

void Cloth::splitEdges(MemoryPool* pool) {
    while (splitSomeEdges(pool));
}

void Cloth::buildAdjacents(std::unordered_map<Node*, std::vector<Edge*>>& adjacentEdges, std::unordered_map<Node*, std::vector<Face*>>& adjacentFaces) const {
    std::vector<Edge*>& edges = mesh->getEdges();
    for (Edge* edge : edges)
        for (int i = 0; i < 2; i++)
            adjacentEdges[edge->nodes[i]].push_back(edge);
            
    std::vector<Face*>& faces = mesh->getFaces();
    for (Face* face : faces)
        for (int i = 0; i < 3; i++)
            adjacentFaces[face->vertices[i]->node].push_back(face);
}

void Cloth::buildAdjacents(thrust::device_vector<int>& edgeBegin, thrust::device_vector<int>& edgeEnd, thrust::device_vector<Edge*>& adjacentEdges, thrust::device_vector<int>& faceBegin, thrust::device_vector<int>& faceEnd, thrust::device_vector<Face*>& adjacentFaces) const {
    int nNodes = mesh->getNodesGpu().size();
    thrust::device_vector<int> indices;

    thrust::device_vector<Edge*>& edges = mesh->getEdgesGpu();
    int nEdges = edges.size();
    edgeBegin.assign(nNodes, 0);
    edgeEnd.assign(nNodes, 0);
    adjacentEdges.resize(2 * nEdges);
    indices.resize(2 * nEdges);
    collectAdjacentEdges<<<GRID_SIZE, BLOCK_SIZE>>>(nEdges, pointer(edges), pointer(indices), pointer(adjacentEdges));
    CUDA_CHECK_LAST();

    thrust::sort_by_key(indices.begin(), indices.end(), adjacentEdges.begin());
    setRange<<<GRID_SIZE, BLOCK_SIZE>>>(2 * nEdges, pointer(indices), pointer(edgeBegin), pointer(edgeEnd));
    CUDA_CHECK_LAST();

    thrust::device_vector<Face*>& faces = mesh->getFacesGpu();
    int nFaces = faces.size();
    faceBegin.assign(nNodes, 0);
    faceEnd.assign(nNodes, 0);
    adjacentFaces.resize(3 * nFaces);
    indices.resize(3 * nFaces);
    collectAdjacentFaces<<<GRID_SIZE, BLOCK_SIZE>>>(nFaces, pointer(faces), pointer(indices), pointer(adjacentFaces));
    CUDA_CHECK_LAST();

    thrust::sort_by_key(indices.begin(), indices.end(), adjacentFaces.begin());
    setRange<<<GRID_SIZE, BLOCK_SIZE>>>(3 * nFaces, pointer(indices), pointer(faceBegin), pointer(faceEnd));
    CUDA_CHECK_LAST();
}

bool Cloth::shouldCollapse(const Edge* edge, int side, const std::unordered_map<Node*, std::vector<Edge*>>& adjacentEdges, const std::unordered_map<Node*, std::vector<Face*>>& adjacentFaces) const {
    Node* node0 = edge->nodes[side];
    Node* node1 = edge->nodes[1 - side];
    if (node0->preserve)
        return false;
    
    bool flag = false;
    const std::vector<Edge*>& edges = adjacentEdges.at(node0);
    for (const Edge* adjacentEdge : edges)
        if (adjacentEdge->isBoundary() || adjacentEdge->isSeam()) {
            flag = true;
            break;
        }
    if (flag && (!edge->isBoundary() && !edge->isSeam()))
        return false;
    
    Vertex* vertex00 = edge->vertices[0][side];
    Vertex* vertex01 = edge->vertices[0][1 - side];
    Vertex* vertex10 = edge->vertices[1][side];
    Vertex* vertex11 = edge->vertices[1][1 - side];
    const std::vector<Face*>& faces = adjacentFaces.at(node0);
    for (const Face* adjacentFace : faces) {
        Vertex* vertices[3] = {adjacentFace->vertices[0], adjacentFace->vertices[1], adjacentFace->vertices[2]};
        if (vertices[0]->node == node1 || vertices[1]->node == node1 || vertices[2]->node == node1)
            continue;
        
        for (int j = 0; j < 3; j++)
            if (vertices[j] == vertex00)
                vertices[j] = vertex01;
            else if (vertices[j] == vertex10)
                vertices[j] = vertex11;
        Vector2f u0 = vertices[0]->u;
        Vector2f u1 = vertices[1]->u;
        Vector2f u2 = vertices[2]->u;
        float a = 0.5f * (u1 - u0).cross(u2 - u0);
        float p = (u0 - u1).norm() + (u1 - u2).norm() + (u2 - u0).norm();
        float aspect = 12.0f * sqrt(3.0f) * a / sqr(p);
        if (a < 1e-6f || aspect < remeshing->aspectMin)
            return false;
        for (int j = 0; j < 3; j++)
            if (vertices[j] != vertex01 && vertices[j] != vertex11 && edgeMetric(vertices[(j + 1) % 3], vertices[(j + 2) % 3]) > 0.9f)
                return false;
    }

    return true;
}

std::vector<PairEi> Cloth::findEdgesToCollapse(const std::unordered_map<Node*, std::vector<Edge*>>& adjacentEdges, const std::unordered_map<Node*, std::vector<Face*>>& adjacentFaces) const {
    std::vector<Edge*>& edges = mesh->getEdges();
    std::unordered_set<Face*> faces;
    std::vector<PairEi> ans;
    for (Edge* edge : edges) {
        int side = -1;
        if (shouldCollapse(edge, 0, adjacentEdges, adjacentFaces))
            side = 0;
        else if (shouldCollapse(edge, 1, adjacentEdges, adjacentFaces))
            side = 1;
        
        if (side > -1) {
            Node* node = edge->nodes[side];
            const std::vector<Face*>& adjacents = adjacentFaces.at(node);
            
            bool flag = true;
            for (Face* adjacentFace : adjacents)
                if (faces.find(adjacentFace) != faces.end()) {
                    flag = false;
                    break;
                }
            
            if (flag) {
                ans.emplace_back(edge, side);
                for (Face* adjacentFace : adjacents)
                    faces.insert(adjacentFace);
            }
        }
    }

    return ans;
}

thrust::device_vector<PairEi> Cloth::findEdgesToCollapse(const thrust::device_vector<int>& edgeBegin, const thrust::device_vector<int>& edgeEnd, const thrust::device_vector<Edge*>& adjacentEdges, const thrust::device_vector<int>& faceBegin, const thrust::device_vector<int>& faceEnd, const thrust::device_vector<Face*>& adjacentFaces) const {
    const int* edgeBeginPointer = pointer(edgeBegin);
    const int* edgeEndPointer = pointer(edgeEnd);
    Edge* const* adjacentEdgesPointer = pointer(adjacentEdges);
    const int* faceBeginPointer = pointer(faceBegin);
    const int* faceEndPointer = pointer(faceEnd);
    Face* const* adjacentFacesPointer = pointer(adjacentFaces);

    thrust::device_vector<Edge*>& edges = mesh->getEdgesGpu();
    int nEdges = edges.size();
    thrust::device_vector<PairEi> edgesToCollapse(nEdges);
    PairEi* edgesToCollapsePointer = pointer(edgesToCollapse);
    thrust::device_vector<float> metrics(nEdges);
    checkEdgesToCollapse<<<GRID_SIZE, BLOCK_SIZE>>>(nEdges, pointer(edges), edgeBeginPointer, edgeEndPointer, adjacentEdgesPointer, faceBeginPointer, faceEndPointer, adjacentFacesPointer, remeshing,edgesToCollapsePointer);
    CUDA_CHECK_LAST();

    edgesToCollapse.erase(thrust::remove_if(edgesToCollapse.begin(), edgesToCollapse.end(), IsNull()), edgesToCollapse.end());
    int nEdgesToCollapse = edgesToCollapse.size();
    thrust::device_vector<PairEi> ans(nEdgesToCollapse, PairEi(nullptr, 0));
    PairEi* ansPointer = pointer(ans);
    initializeCollapseFaces<<<GRID_SIZE, BLOCK_SIZE>>>(nEdgesToCollapse, edgesToCollapsePointer, faceBeginPointer, faceEndPointer, adjacentFacesPointer);
    CUDA_CHECK_LAST();

    int num, newNum = nEdgesToCollapse;
    do {
        num = newNum;
        resetCollapseFaces<<<GRID_SIZE, BLOCK_SIZE>>>(nEdgesToCollapse, edgesToCollapsePointer, faceBeginPointer, faceEndPointer, adjacentFacesPointer);
        CUDA_CHECK_LAST();

        computeCollapseMinIndices<<<GRID_SIZE, BLOCK_SIZE>>>(nEdgesToCollapse, edgesToCollapsePointer, faceBeginPointer, faceEndPointer, adjacentFacesPointer);
        CUDA_CHECK_LAST();

        checkIndependentEdgesToCollapse<<<GRID_SIZE, BLOCK_SIZE>>>(nEdgesToCollapse, edgesToCollapsePointer, faceBeginPointer, faceEndPointer, adjacentFacesPointer, ansPointer);
        CUDA_CHECK_LAST();

        newNum = thrust::count_if(ans.begin(), ans.end(), IsNull());
    } while (num > newNum);

    ans.erase(thrust::remove_if(ans.begin(), ans.end(), IsNull()), ans.end());
    return ans;
}

bool Cloth::collapseSomeEdges(MemoryPool* pool) {
    Operator op;

    if (!gpu) {
        std::unordered_map<Node*, std::vector<Edge*>> adjacentEdges;
        std::unordered_map<Node*, std::vector<Face*>> adjacentFaces;
        buildAdjacents(adjacentEdges, adjacentFaces);

        std::vector<PairEi> edgesToCollapse = std::move(findEdgesToCollapse(adjacentEdges, adjacentFaces));
        if (edgesToCollapse.empty())
            return false;

        for (const PairEi& edge : edgesToCollapse)
            op.collapse(edge.first, edge.second, material, adjacentEdges, adjacentFaces, pool);
    } else {
        mesh->updateIndices();

        thrust::device_vector<int> edgeBegin, edgeEnd, faceBegin, faceEnd;
        thrust::device_vector<Edge*> adjacentEdges;
        thrust::device_vector<Face*> adjacentFaces;
        buildAdjacents(edgeBegin, edgeEnd, adjacentEdges, faceBegin, faceEnd, adjacentFaces);

        thrust::device_vector<PairEi> edgesToCollapse = std::move(findEdgesToCollapse(edgeBegin, edgeEnd, adjacentEdges, faceBegin, faceEnd, adjacentFaces));
        if (edgesToCollapse.empty())
            return false;

        op.collapse(edgesToCollapse, material, edgeBegin, edgeEnd, adjacentEdges, faceBegin, faceEnd, adjacentFaces, pool);
    }

    mesh->apply(op);
    flipEdges(pool);
    return true;
}

void Cloth::collapseEdges(MemoryPool* pool) {
    while (collapseSomeEdges(pool));
}

Mesh* Cloth::getMesh() const {
    return mesh;
}

void Cloth::physicsStep(float dt, const Vector3f& gravity, const Wind* wind, float handleStiffness, const std::vector<Proximity>& proximities, float repulsionThickness) {
    Eigen::SparseMatrix<float> A;
    Eigen::VectorXf b;

    initializeForces(A, b);
    addExternalForces(dt, gravity, wind, A, b);
    addInternalForces(dt, A, b);
    addHandleForces(dt, handleStiffness, A, b);
    addProximityForces(dt, proximities, repulsionThickness, A, b);

    Eigen::SimplicialLLT<Eigen::SparseMatrix<float>> cholesky;
    cholesky.compute(A);
    Eigen::VectorXf dv = cholesky.solve(b);

    std::vector<Node*>& nodes = mesh->getNodes();
    for (int i = 0; i < nodes.size(); i++)
        nodes[i]->v += Vector3f(dv(3 * i), dv(3 * i + 1), dv(3 * i + 2));
}

void Cloth::physicsStep(float dt, const Vector3f& gravity, const Wind* wind, float handleStiffness, const thrust::device_vector<Proximity>& proximities, float repulsionThickness) {
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
    int nProximities = proximities.size();

    int aSize = 3 * nNodes + 144 * nEdges + 81 * nFaces + 3 * nHandles + 288 * nProximities;
    int bSize = 3 * nNodes + 12 * nEdges + 18 * nFaces + 3 * nHandles + 24 * nProximities;

    thrust::device_vector<Pairii> aIndices(aSize);
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

    addProximityForcesGpu<<<GRID_SIZE, BLOCK_SIZE>>>(nProximities, pointer(proximities), dt, repulsionThickness, nNodes, nodesPointer, pointer(aIndices, 3 * nNodes + 144 * nEdges + 81 * nFaces + 3 * nHandles), pointer(aValues, 3 * nNodes + 144 * nEdges + 81 * nFaces + 3 * nHandles), pointer(bIndices, 3 * nNodes + 12 * nEdges + 18 * nFaces + 3 * nHandles), pointer(bValues, 3 * nNodes + 12 * nEdges + 18 * nFaces + 3 * nHandles));
    CUDA_CHECK_LAST();

    int n = 3 * nNodes;

    thrust::sort_by_key(aIndices.begin(), aIndices.end(), aValues.begin());
    thrust::device_vector<Pairii> outputAIndices(aSize);
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
    CUSOLVER_CHECK(cusolverSpScsrlsvchol(cusolverHandle, n, nNonZero, descr, pointer(values), pointer(rowPointer), pointer(colIndies), pointer(b), 1e-10f, 0, pointer(dv), &singularity));
    if (singularity != -1)
        std::cerr << "Not SPD! " << std::endl;

    updateNodes<<<GRID_SIZE, BLOCK_SIZE>>>(nNodes, dt, pointer(dv), nodesPointer);
    CUDA_CHECK_LAST();
}

void Cloth::remeshingStep(const std::vector<BVH*>& obstacleBvhs, float thickness, MemoryPool* pool) {
    if (!gpu) {
        std::vector<Plane> planes = std::move(findNearestPlane(obstacleBvhs, thickness));
        computeSizing(planes);

        flipEdges(pool);
        splitEdges(pool);
        collapseEdges(pool);

        mesh->updateIndices();
    } else {
        thrust::device_vector<Plane> planes = std::move(findNearestPlaneGpu(obstacleBvhs, thickness));
        computeSizing(planes);

        flipEdges(pool);
        splitEdges(pool);
        collapseEdges(pool);
    }
}

void Cloth::bind() {
    edgeShader = new Shader("shader/Vertex.glsl", "shader/EdgeFragment.glsl");
    faceShader = new Shader("shader/Vertex.glsl", "shader/FaceFragment.glsl");
    mesh->bind();
}

void Cloth::render(const Matrix4x4f& model, const Matrix4x4f& view, const Matrix4x4f& projection, const Vector3f& cameraPosition, const Vector3f& lightDirection) const {
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
    mesh->render();
}

void Cloth::load(const std::string& path) {
    Transformation transformation(Json::nullValue);
    mesh->load(path, transformation, material, nullptr);
}

void Cloth::save(const std::string& path, Json::Value& json) const {
    json["mesh"] = path;
    json["transform"] = Json::nullValue;
    if (!gpu) {
        int index = 0;
        for (Json::Value& handleJson : json["handles"])
            for (Json::Value& nodeJson : handleJson["nodes"])
                nodeJson = handles[index++].node->index;
    } else {
        int nHandles = handlesGpu.size();
        thrust::device_vector<int> handleIndices(nHandles);
        collectHandleIndices<<<GRID_SIZE, BLOCK_SIZE>>>(nHandles, pointer(handlesGpu), pointer(handleIndices));
        CUDA_CHECK_LAST();

        int index = 0;
        for (Json::Value& handleJson : json["handles"])
            for (Json::Value& nodeJson : handleJson["nodes"]) {
                int handleIndex = handleIndices[index++];
                nodeJson = handleIndex;
            }
    }
    mesh->save(path);
}