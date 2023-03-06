#include "Cloth.cuh"

Cloth::Cloth(const Json::Value& json) {
    Transform* transform = new Transform(json["transform"]);
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
    mesh = new Mesh(json["mesh"], transform, material);

    std::vector<int> handleIndices;
    for (const Json::Value& handleJson : json["handles"])
        for (const Json::Value& nodeJson : handleJson["nodes"])
            handleIndices.push_back(parseInt(nodeJson));
    
    edgeShader = new Shader("shader/Vertex.glsl", "shader/EdgeFragment.glsl");
    faceShader = new Shader("shader/Vertex.glsl", "shader/FaceFragment.glsl");
    delete transform;

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
    thrust::device_vector<Face*> faces = mesh->getFacesGpu();
    int nFaces = faces.size();
    thrust::device_vector<int> indices(3 * nFaces);
    thrust::device_vector<Pairfm> sizing(3 * nFaces);
    collectVertexSizing<<<GRID_SIZE, BLOCK_SIZE>>>(faces.size(), pointer(faces), pointer(planes), pointer(indices), pointer(sizing), remeshing);
    CUDA_CHECK_LAST();

    thrust::sort_by_key(indices.begin(), indices.end(), sizing.begin());
    thrust::device_vector<int> outputIndices(3 * nFaces);
    thrust::device_vector<Pairfm> outputSizing(3 * nFaces);
    auto iter = thrust::reduce_by_key(indices.begin(), indices.end(), sizing.begin(), outputIndices.begin(), outputSizing.begin());
    thrust::device_vector<Vertex*> vertices = mesh->getVerticesGpu();
    setVertexSizing<<<GRID_SIZE, BLOCK_SIZE>>>(iter.first - outputIndices.begin(), pointer(outputIndices), pointer(outputSizing), pointer(vertices));
    CUDA_CHECK_LAST();
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
    if (edge->isBoundary() || edge->isSeam())
        return false;
        
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

std::vector<Edge*> Cloth::findEdgesToFlip() const {
    std::vector<Edge*>& edges = mesh->getEdges();
    std::unordered_set<Node*> nodes;
    std::vector<Edge*> ans;
    for (Edge* edge : edges)
        if (shouldFlip(edge)) {
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

bool Cloth::flipSomeEdges() {
    static int nEdges = 0;
    std::vector<Edge*> edgesToFlip = std::move(findEdgesToFlip());
    if (edgesToFlip.size() == nEdges)
        return false;
    
    nEdges = edgesToFlip.size();
    Operator op;
    for (const Edge* edge : edgesToFlip)
        op.flip(edge, material);

    mesh->apply(op);
    return !edgesToFlip.empty();
}

void Cloth::flipEdges() {
    int nEdges = mesh->getEdges().size();
    for (int i = 0; i < 2 * nEdges; i++)
        if (!flipSomeEdges())
            return;
}

std::vector<Edge*> Cloth::findEdgesToSplit() const {
    std::vector<Edge*>& edges = mesh->getEdges();
    std::vector<Pairfe> sorted;
    for (Edge* edge : edges) {
        float m = edgeMetric(edge);
        if (m > 1.0f)
            sorted.emplace_back(m, edge);
    }
    std::sort(sorted.begin(), sorted.end(), [](const Pairfe& a, const Pairfe& b) {
        return a.first > b.first;
    });

    std::unordered_set<Node*> nodes;
    std::vector<Edge*> ans;
    for (const Pairfe& p : sorted) {
        Edge* edge = p.second;
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

bool Cloth::splitSomeEdges() {
    std::vector<Edge*> edgesToSplit = std::move(findEdgesToSplit());
    Operator op;
    int nNodes = mesh->getNodes().size();
    for (const Edge* edge : edgesToSplit)
        op.split(edge, material, nNodes++);

    mesh->apply(op);
    flipEdges();
    return !edgesToSplit.empty();
}

void Cloth::splitEdges() {
    while (splitSomeEdges());
}

void Cloth::buildAdjacents(std::unordered_map<Node*, std::vector<Edge*>>& adjacentEdges, std::unordered_map<Vertex*, std::vector<Face*>>& adjacentFaces) const {
    std::vector<Edge*>& edges = mesh->getEdges();
    for (Edge* edge : edges)
        for (int i = 0; i < 2; i++)
            adjacentEdges[edge->nodes[i]].push_back(edge);
            
    std::vector<Face*>& faces = mesh->getFaces();
    for (Face* face : faces)
        for (int i = 0; i < 3; i++)
            adjacentFaces[face->vertices[i]].push_back(face);
}

bool Cloth::shouldCollapse(const Edge* edge, int side, const std::unordered_map<Node*, std::vector<Edge*>>& adjacentEdges, const std::unordered_map<Vertex*, std::vector<Face*>>& adjacentFaces) const {
    Node* node = edge->nodes[side];
    if (node->preserve)
        return false;
    
    bool flag = false;
    const std::vector<Edge*>& edges = adjacentEdges.at(node);
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
            
            const std::vector<Face*>& faces = adjacentFaces.at(vertex0);
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

        const std::vector<Face*>& faces = adjacentFaces.at(vertex0);
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

std::vector<Pairei> Cloth::findEdgesToCollapse(const std::unordered_map<Node*, std::vector<Edge*>>& adjacentEdges, const std::unordered_map<Vertex*, std::vector<Face*>>& adjacentFaces) const {
    std::vector<Edge*>& edges = mesh->getEdges();
    std::unordered_set<Node*> nodes;
    std::vector<Pairei> ans;
    for (Edge* edge : edges) {
        int side = -1;
        if (shouldCollapse(edge, 0, adjacentEdges, adjacentFaces))
            side = 0;
        else if (shouldCollapse(edge, 1, adjacentEdges, adjacentFaces))
            side = 1;
        
        if (side > -1) {
            Node* node = edge->nodes[side];
            if (nodes.find(node) == nodes.end()) {
                bool flag = true;
                const std::vector<Edge*>& adjacents = adjacentEdges.at(node);
                for (const Edge* adjacentEdge : adjacents) {
                    Node* adjacentNode = adjacentEdge->nodes[0] != node ? adjacentEdge->nodes[0] : adjacentEdge->nodes[1];
                    if (nodes.find(adjacentNode) != nodes.end()) {
                        flag = false;
                        break;
                    }
                }

                if (flag) {
                    ans.emplace_back(edge, side);
                    nodes.insert(node);
                    for (const Edge* adjacentEdge : adjacents) {
                        Node* adjacentNode = adjacentEdge->nodes[0] != node ? adjacentEdge->nodes[0] : adjacentEdge->nodes[1];
                        nodes.insert(adjacentNode);
                    }
                }
            }
        }
    }

    return ans;
}

bool Cloth::collapseSomeEdges() {
    std::unordered_map<Node*, std::vector<Edge*>> adjacentEdges;
    std::unordered_map<Vertex*, std::vector<Face*>> adjacentFaces;
    buildAdjacents(adjacentEdges, adjacentFaces);

    std::vector<Pairei> edgesToCollapse = std::move(findEdgesToCollapse(adjacentEdges, adjacentFaces));
    Operator op;
    for (const Pairei& edge : edgesToCollapse)
        op.collapse(edge.first, edge.second, material, adjacentEdges, adjacentFaces);

    mesh->apply(op);
    flipEdges();
    return !edgesToCollapse.empty();
}

void Cloth::collapseEdges() {
    while (collapseSomeEdges());
}

Mesh* Cloth::getMesh() const {
    return mesh;
}

void Cloth::readDataFromFile(const std::string& path) {
    mesh->readDataFromFile(path);
    mesh->updateNodeGeometries();
    mesh->updateFaceGeometries();
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
        CUSOLVER_CHECK(cusolverSpScsrlsvchol(cusolverHandle, n, nNonZero, descr, pointer(values), pointer(rowPointer), pointer(colIndies), pointer(b), 1e-5f, 0, pointer(dv), &singularity));
        if (singularity != -1)
            std::cerr << "Not SPD! " << std::endl;

        updateNodes<<<GRID_SIZE, BLOCK_SIZE>>>(nNodes, dt, pointer(dv), nodesPointer);
        CUDA_CHECK_LAST();
    }
}

void Cloth::remeshingStep(const std::vector<BVH*>& obstacleBvhs, float thickness) {
    if (!gpu) {
        std::vector<Plane> planes = std::move(findNearestPlane(obstacleBvhs, thickness));
        computeSizing(planes);

        flipEdges();
        splitEdges();
        collapseEdges();
    } else {
        thrust::device_vector<Plane> planes = std::move(findNearestPlaneGpu(obstacleBvhs, thickness));
        computeSizing(planes);

        // TODO
    }
}

void Cloth::updateStructures() {
    mesh->updateStructures();
}

void Cloth::updateNodeGeometries() {
    mesh->updateNodeGeometries();
}

void Cloth::updateFaceGeometries() {
    mesh->updateFaceGeometries();
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