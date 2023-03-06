#include "Mesh.cuh"

Mesh::Mesh(const Json::Value &json, const Transform* transform, const Material* material) {
    std::ifstream fin(json.asString());
    if (!fin.is_open()) {
        std::cerr << "Failed to open mesh file: " << json.asString() << std::endl;
        exit(1);
    }

    std::string line;
    std::vector<Vector3f> x;
    std::vector<Vector2f> u;
    std::vector<int> xIndices, uIndices;
    while (getline(fin, line)) {
        std::vector<std::string> s = std::move(split(line, ' '));
        if (s[0] == "v")
            x.push_back(transform->applyTo(Vector3f(std::stod(s[1]), std::stod(s[2]), std::stod(s[3]))));
        else if (s[0] == "vt")
            u.emplace_back(std::stof(s[1]), std::stof(s[2]));
        else if (s[0] == "f")
            for (int i = 1; i < 4; i++)
                if (line.find('/') != std::string::npos) {
                    std::vector<std::string> t = std::move(split(s[i], '/'));
                    xIndices.push_back(std::stoi(t[0]) - 1);
                    uIndices.push_back(std::stoi(t[1]) - 1);
                } else {
                    u.emplace_back(0.0f, 0.0f);
                    xIndices.push_back(std::stoi(s[i]) - 1);
                    uIndices.push_back(u.size() - 1);
                }
    }
    fin.close();

    initialize(x, u, xIndices, uIndices, material);
}

Mesh::Mesh(const Mesh* mesh) {
    nodes.resize(mesh->nodes.size());
    for (int i = 0; i < mesh->nodes.size(); i++) {
        Node* node = mesh->nodes[i];
        nodes[i] = new Node(node->x, node->isFree);
    }

    vertices.resize(mesh->vertices.size());
    for (int i = 0; i < mesh->vertices.size(); i++) {
        Vertex* vertex = mesh->vertices[i];
        vertices[i] = new Vertex(vertex->u);
        vertices[i]->node = nodes[vertex->node->index];
    }
    
    edges.resize(mesh->edges.size());
    for (int i = 0; i < mesh->edges.size(); i++) {
        Edge* edge = mesh->edges[i];
        edges[i] = new Edge(nodes[edge->nodes[0]->index], nodes[edge->nodes[1]->index]);
    }
    
    faces.resize(mesh->faces.size());
    for (int i = 0; i < mesh->faces.size(); i++) {
        Face* face = mesh->faces[i];
        faces[i] = new Face(vertices[face->vertices[0]->index], vertices[face->vertices[1]->index], vertices[face->vertices[2]->index], nullptr);
    }
}

Mesh::~Mesh() {
    if (!gpu) {
        for (const Node* node : nodes)
            delete node;
        for (const Vertex* vertex : vertices)
            delete vertex;
        for (const Edge* edge : edges)
            delete edge;
        for (const Face* face : faces)
            delete face;
    } else {
        deleteNodes<<<GRID_SIZE, BLOCK_SIZE>>>(nodesGpu.size(), pointer(nodesGpu));
        CUDA_CHECK_LAST();
        deleteVertices<<<GRID_SIZE, BLOCK_SIZE>>>(verticesGpu.size(), pointer(verticesGpu));
        CUDA_CHECK_LAST();
        deleteEdges<<<GRID_SIZE, BLOCK_SIZE>>>(edgesGpu.size(), pointer(edgesGpu));
        CUDA_CHECK_LAST();
        deleteFaces<<<GRID_SIZE, BLOCK_SIZE>>>(facesGpu.size(), pointer(facesGpu));
        CUDA_CHECK_LAST();
    }
}

std::vector<std::string> Mesh::split(const std::string& s, char c) const {
    std::string t = s;
    std::vector<std::string> ans;
    while (t.find(c) != std::string::npos) {
        int index = t.find(c);
        ans.push_back(t.substr(0, index));
        t.erase(0, index + 1);
    }
    ans.push_back(t);
    return ans;
}

Edge* Mesh::findEdge(int index0, int index1, std::map<Pairii, int>& edgeMap) {
    if (index0 > index1)
        mySwap(index0, index1);
    Pairii index(index0, index1);
    auto iter = edgeMap.find(index);
    if (iter != edgeMap.end())
        return edges[iter->second];
    else {
        edgeMap[index] = edges.size();
        edges.push_back(new Edge(nodes[index0], nodes[index1]));
        return edges.back();
    }
}

void Mesh::initialize(const std::vector<Vector3f>& x, const std::vector<Vector2f>& u, const std::vector<int>& xIndices, const std::vector<int>& uIndices, const Material* material) {
    bool isFree = (material != nullptr);
    if (!gpu) {
        nodes.resize(x.size());
        vertices.resize(u.size());
        faces.resize(xIndices.size() / 3);
        for (int i = 0; i < x.size(); i++)
            nodes[i] = new Node(x[i], isFree);
        for (int i = 0; i < u.size(); i++)
            vertices[i] = new Vertex(u[i]);

        std::map<Pairii, int> edgeMap;
        for (int i = 0; i < xIndices.size(); i += 3) {
            int xIndex0 = xIndices[i];
            int xIndex1 = xIndices[i + 1];
            int xIndex2 = xIndices[i + 2];
            int uIndex0 = uIndices[i];
            int uIndex1 = uIndices[i + 1];
            int uIndex2 = uIndices[i + 2];

            Vertex* vertex0 = vertices[uIndex0];
            Vertex* vertex1 = vertices[uIndex1];
            Vertex* vertex2 = vertices[uIndex2];
            
            Edge* edge0 = findEdge(xIndex0, xIndex1, edgeMap);
            Edge* edge1 = findEdge(xIndex1, xIndex2, edgeMap);
            Edge* edge2 = findEdge(xIndex2, xIndex0, edgeMap);
            Face* face = new Face(vertex0, vertex1, vertex2, material);

            vertex0->node = nodes[xIndex0];
            vertex1->node = nodes[xIndex1];
            vertex2->node = nodes[xIndex2];
            edge0->initialize(vertex2, face);
            edge1->initialize(vertex0, face);
            edge2->initialize(vertex1, face);
            face->setEdges(edge0, edge1, edge2);

            faces[i / 3] = face;
        }

        for (const Edge* edge : edges)
            if (edge->isBoundary() || edge->isSeam())
                for (int i = 0; i < 2; i++)
                    edge->nodes[i]->preserve = true;
    } else {
        int nNodes = x.size();
        int nVertices = u.size();
        int nFaces = xIndices.size() / 3;
        int nEdges = xIndices.size();
        thrust::device_vector<Vector3f> xGpu = x;
        thrust::device_vector<Vector2f> uGpu = u;
        thrust::device_vector<int> xIndicesGpu = xIndices;
        thrust::device_vector<int> uIndicesGpu = uIndices;

        nodesGpu.resize(nNodes);
        Node** nodesPointer = pointer(nodesGpu);
        initializeNodes<<<GRID_SIZE, BLOCK_SIZE>>>(nNodes, pointer(xGpu), isFree, nodesPointer);
        CUDA_CHECK_LAST();

        verticesGpu.resize(nVertices);
        Vertex** verticesPointer = pointer(verticesGpu);
        initializeVertices<<<GRID_SIZE, BLOCK_SIZE>>>(nVertices, pointer(uGpu), verticesPointer);
        CUDA_CHECK_LAST();

        facesGpu.resize(nFaces);
        Face** facesPointer = pointer(facesGpu);
        thrust::device_vector<Pairii> edgeIndices(nEdges);
        Pairii* edgeIndicesPointer = pointer(edgeIndices);
        thrust::device_vector<EdgeData> edgeData(nEdges);
        EdgeData* edgeDataPointer = pointer(edgeData);
        initializeFaces<<<GRID_SIZE, BLOCK_SIZE>>>(nFaces, pointer(xIndicesGpu), pointer(uIndicesGpu), nodesPointer, material, verticesPointer, facesPointer, edgeIndicesPointer, edgeDataPointer);
        CUDA_CHECK_LAST();
        thrust::sort_by_key(edgeIndices.begin(), edgeIndices.end(), edgeData.begin());

        edgesGpu.resize(nEdges);
        Edge** edgesPointer = pointer(edgesGpu);
        initializeEdges<<<GRID_SIZE, BLOCK_SIZE>>>(nEdges, edgeIndicesPointer, edgeDataPointer, nodesPointer, edgesPointer);
        CUDA_CHECK_LAST();
        setEdges<<<GRID_SIZE, BLOCK_SIZE>>>(nEdges, edgeIndicesPointer, edgeDataPointer, edgesPointer);
        CUDA_CHECK_LAST();
        edgesGpu.erase(thrust::remove_if(edgesGpu.begin(), edgesGpu.end(), IsNull()), edgesGpu.end());

        nEdges = edgesGpu.size();
        thrust::device_vector<int> nodeIndices(2 * nEdges);
        int* nodeIndicesPointer = pointer(nodeIndices);
        collectPreservedNodes<<<GRID_SIZE, BLOCK_SIZE>>>(nEdges, edgesPointer, nodeIndicesPointer);
        CUDA_CHECK_LAST();
        thrust::sort(nodeIndices.begin(), nodeIndices.end());
        auto iter = thrust::unique(nodeIndices.begin(), nodeIndices.end());
        setPreservedNodes<<<GRID_SIZE, BLOCK_SIZE>>>(iter - nodeIndices.begin(), nodeIndicesPointer, nodesPointer);
        CUDA_CHECK_LAST();
    }

    updateStructures();
    updateNodeGeometries();
    updateFaceGeometries();
}

std::vector<Node*>& Mesh::getNodes() {
    return nodes;
}

thrust::device_vector<Node*>& Mesh::getNodesGpu() {
    return nodesGpu;
}

std::vector<Vertex*>& Mesh::getVertices() {
    return vertices;
}

thrust::device_vector<Vertex*>& Mesh::getVerticesGpu() {
    return verticesGpu;
}

std::vector<Edge*>& Mesh::getEdges() {
    return edges;
}

thrust::device_vector<Edge*>& Mesh::getEdgesGpu() {
    return edgesGpu;
}

std::vector<Face*>& Mesh::getFaces() {
    return faces;
}

thrust::device_vector<Face*>& Mesh::getFacesGpu() {
    return facesGpu;
}

bool Mesh::contain(const Vertex* vertex) const {
    int index = vertex->index;
    return index < vertices.size() && vertices[index] == vertex;
}

bool Mesh::contain(const Face* face) const {
    return contain(face->vertices[0]) && contain(face->vertices[1]) && contain(face->vertices[2]);
}

void Mesh::reset() {
    if (!gpu)
        for (Node* node : nodes)
            node->x = node->x0;
    else {
        resetGpu<<<GRID_SIZE, BLOCK_SIZE>>>(nodesGpu.size(), pointer(nodesGpu));
        CUDA_CHECK_LAST();
    }
}

Vector3f Mesh::oldPosition(const Vector2f& u) const {
    for (const Face* face : faces) {
        Vector3f b = face->barycentricCoordinates(u);
        if (b(0) >= -1e-6f && b(1) >= -1e-6f && b(2) >= -1e-5f)
            return face->position(b);
    }
}

void Mesh::apply(const Operator& op) {
    for (const Node* node : op.removedNodes)
        nodes.erase(std::remove(nodes.begin(), nodes.end(), node), nodes.end());
    nodes.insert(nodes.end(), op.addedNodes.begin(), op.addedNodes.end());

    for (const Vertex* vertex : op.removedVertices)
        vertices.erase(std::remove(vertices.begin(), vertices.end(), vertex), vertices.end());
    vertices.insert(vertices.end(), op.addedVertices.begin(), op.addedVertices.end());
    
    for (const Edge* edge : op.removedEdges)
        edges.erase(std::remove(edges.begin(), edges.end(), edge), edges.end());
    edges.insert(edges.end(), op.addedEdges.begin(), op.addedEdges.end());

    for (const Face* face : op.removedFaces)
        faces.erase(std::remove(faces.begin(), faces.end(), face), faces.end());
    faces.insert(faces.end(), op.addedFaces.begin(), op.addedFaces.end());

    for (const Node* node : op.removedNodes)
        delete node;
    for (const Vertex* vertex : op.removedVertices)
        delete vertex;
    for (const Edge* edge : op.removedEdges)
        delete edge;
    for (const Face* face : op.removedFaces)
        delete face;
}

void Mesh::updateStructures() {
    if (!gpu) {
        for (int i = 0; i < nodes.size(); i++) {
            Node* node = nodes[i];
            node->index = i;
            node->mass = 0.0f;
            node->area = 0.0f;
        }
        for (int i = 0; i < vertices.size(); i++)
            vertices[i]->index = i;
        for (const Face* face : faces) {
            float mass = face->mass / 3.0f;
            float area = face->area;
            for (int i = 0; i < 3; i++) {
                Node* node = face->vertices[i]->node;
                node->mass += mass;
                node->area += area;
            }
        }
    } else {
        updateNodeIndices<<<GRID_SIZE, BLOCK_SIZE>>>(nodesGpu.size(), pointer(nodesGpu));
        CUDA_CHECK_LAST();
        updateVertexIndices<<<GRID_SIZE, BLOCK_SIZE>>>(verticesGpu.size(), pointer(verticesGpu));
        CUDA_CHECK_LAST();

        thrust::device_vector<int> indices(3 * facesGpu.size());
        thrust::device_vector<NodeData> nodeData(3 * facesGpu.size());
        collectNodeStructures<<<GRID_SIZE, BLOCK_SIZE>>>(facesGpu.size(), pointer(facesGpu), pointer(indices), pointer(nodeData));
        CUDA_CHECK_LAST();
        
        thrust::sort_by_key(indices.begin(), indices.end(), nodeData.begin());
        thrust::device_vector<int> outputIndices(3 * facesGpu.size());
        thrust::device_vector<NodeData> outputNodeData(3 * facesGpu.size());
        auto iter = thrust::reduce_by_key(indices.begin(), indices.end(), nodeData.begin(), outputIndices.begin(), outputNodeData.begin());
        setNodeStructures<<<GRID_SIZE, BLOCK_SIZE>>>(iter.first - outputIndices.begin(), pointer(outputIndices), pointer(outputNodeData), pointer(nodesGpu));
        CUDA_CHECK_LAST();
    }
}

void Mesh::updateNodeGeometries() {
    if (!gpu) {
        for (Node* node : nodes) {
            node->x1 = node->x;
            node->n = Vector3f();
        }
        for (const Face* face : faces)
            for (int i = 0; i < 3; i++) {
                Node* node = face->vertices[i]->node;
                Vector3f e0 = face->vertices[(i + 1) % 3]->node->x - node->x;
                Vector3f e1 = face->vertices[(i + 2) % 3]->node->x - node->x;
                node->n += e0.cross(e1) / (e0.norm2() * e1.norm2());
            }
        for (Node* node : nodes)
            node->n.normalize();
    } else {
        updateNodeGeometriesGpu<<<GRID_SIZE, BLOCK_SIZE>>>(nodesGpu.size(), pointer(nodesGpu));
        CUDA_CHECK_LAST();

        thrust::device_vector<int> indices(3 * facesGpu.size());
        thrust::device_vector<Vector3f> nodeData(3 * facesGpu.size());
        collectNodeGeometries<<<GRID_SIZE, BLOCK_SIZE>>>(facesGpu.size(), pointer(facesGpu), pointer(indices), pointer(nodeData));
        CUDA_CHECK_LAST();

        thrust::sort_by_key(indices.begin(), indices.end(), nodeData.begin());
        thrust::device_vector<int> outputIndices(3 * facesGpu.size());
        thrust::device_vector<Vector3f> outputNodeData(3 * facesGpu.size());
        auto iter = thrust::reduce_by_key(indices.begin(), indices.end(), nodeData.begin(), outputIndices.begin(), outputNodeData.begin());
        setNodeGeometries<<<GRID_SIZE, BLOCK_SIZE>>>(iter.first - outputIndices.begin(), pointer(outputIndices), pointer(outputNodeData), pointer(nodesGpu));
        CUDA_CHECK_LAST();
    }
}

void Mesh::updateFaceGeometries() {
    if (!gpu)
        for (Face* face : faces)
            face->update();
    else {
        updateFaceGeometriesGpu<<<GRID_SIZE, BLOCK_SIZE>>>(facesGpu.size(), pointer(facesGpu));
        CUDA_CHECK_LAST();
    }
}

void Mesh::updateVelocities(float dt) {
    float invDt = 1.0f / dt;
    if (!gpu)
        for (Node* node : nodes)
            node->v = (node->x - node->x0) * invDt;
    else {
        updateVelocitiesGpu<<<GRID_SIZE, BLOCK_SIZE>>>(nodesGpu.size(), invDt, pointer(nodesGpu));
        CUDA_CHECK_LAST();
    }
}

void Mesh::updateRenderingData(bool rebind) {
    if (!gpu) {
        std::vector<Renderable> renderables(3 * faces.size());
        for (int i = 0; i < faces.size(); i++) {
            Face* face = faces[i];
            for (int j = 0; j < 3; j++) {
                Vertex* vertex = face->vertices[j];
                Node* node = vertex->node;
                int index = 3 * i + j;
                renderables[index].x = node->x;
                renderables[index].n = node->n;
                renderables[index].u = vertex->u;
            }
        }

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, renderables.size() * sizeof(Renderable), renderables.data(), GL_DYNAMIC_DRAW);
    } else {
        if (rebind) {
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, 3 * facesGpu.size() * sizeof(Renderable), nullptr, GL_DYNAMIC_DRAW);

            CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&vboCuda, vbo, cudaGraphicsRegisterFlagsWriteDiscard));
        }

        CUDA_CHECK(cudaGraphicsMapResources(1, &vboCuda));
        Renderable* renderables;
        size_t nRenderanles;
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&renderables), &nRenderanles, vboCuda));
        updateRenderingDataGpu<<<GRID_SIZE, BLOCK_SIZE>>>(facesGpu.size(), pointer(facesGpu), renderables);
        CUDA_CHECK_LAST();

        CUDA_CHECK(cudaGraphicsUnmapResources(1, &vboCuda));
    }
}

void Mesh::bind() {
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Renderable), reinterpret_cast<void*>(offsetof(Renderable, x)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Renderable), reinterpret_cast<void*>(offsetof(Renderable, n)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Renderable), reinterpret_cast<void*>(offsetof(Renderable, u)));

    updateRenderingData(true);
}

void Mesh::render() const {
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3 * (!gpu ? faces.size() : facesGpu.size()));
    glBindVertexArray(0);
}

void Mesh::readDataFromFile(const std::string& path) {
    std::ifstream fin(path);
    for (Node* node : nodes)
        fin >> node->x0(0) >> node->x0(1) >> node->x0(2) >> node->x(0) >> node->x(1) >> node->x(2) >> node->v(0) >> node->v(1) >> node->v(2);
    fin.close();
}

void Mesh::writeDataToFile(const std::string& path) {
    std::ofstream fout(path);
    if (!gpu) {
        for (const Node* node : nodes) {
            fout << "v " << node->x(0) << " " << node->x(1) << " " << node->x(2) << std::endl;
            fout << "nv " << node->v(0) << " " << node->v(1) << " " << node->v(2) << std::endl;
        }
        for (const Vertex* vertex : vertices)
            fout << "vt " << vertex->u(0) << " " << vertex->u(1) << std::endl;
        for (const Face* face : faces) {
            fout << "f";
            for (int i = 0; i < 3; i++) {
                Vertex* vertex = face->vertices[i];
                int xIndex = vertex->node->index + 1;
                int uIndex = vertex->index + 1;
                fout << " " << xIndex << "/" << uIndex;
            }
            fout << std::endl;
        }
    } else {
        int nNodes = nodesGpu.size();
        thrust::device_vector<Vector3f> x(nNodes);
        copyX<<<GRID_SIZE, BLOCK_SIZE>>>(nNodes, pointer(nodesGpu), pointer(x));
        CUDA_CHECK_LAST();
        for (const Vector3f& xt : x)
            fout << "v " << xt(0) << " " << xt(1) << " " << xt(2) << std::endl;

        thrust::device_vector<Vector3f> v(nNodes);
        copyX<<<GRID_SIZE, BLOCK_SIZE>>>(nNodes, pointer(nodesGpu), pointer(v));
        CUDA_CHECK_LAST();
        for (const Vector3f& vt : v)
            fout << "nv " << vt(0) << " " << vt(1) << " " << vt(2) << std::endl;

        int nVertices = verticesGpu.size();
        thrust::device_vector<Vector2f> u(nVertices);
        copyU<<<GRID_SIZE, BLOCK_SIZE>>>(nVertices, pointer(verticesGpu), pointer(u));
        CUDA_CHECK_LAST();
        for (const Vector2f& ut : u)
            fout << "vt " << ut(0) << " " << ut(1) << std::endl;
        
        int nFaces = facesGpu.size();
        thrust::device_vector<Pairii> indices(3 * nFaces);
        copyFaceIndices<<<GRID_SIZE, BLOCK_SIZE>>>(nFaces, pointer(facesGpu), pointer(indices));
        CUDA_CHECK_LAST();
        for (int i = 0; i < nFaces; i++) {
            fout << "f";
            for (int j = 0; j < 3; j++) {
                const Pairii& index = indices[3 * i + j];
                fout << " " << index.first + 1 << "/" << index.second + 1;
            }
            fout << std::endl;
        }

        fout.close();
    }
}

void Mesh::printDebugInfo(int selectedFace) {
    if (!gpu) {
        Face* face = faces[selectedFace];
        std::cout << "Nodes=[" << face->vertices[0]->node->index << ", " << face->vertices[1]->node->index << ", " << face->vertices[2]->node->index << "]" << std::endl;
    } else
        printDebugInfoGpu<<<1, 1>>>(pointer(facesGpu), selectedFace);
}