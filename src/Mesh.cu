#include "Mesh.cuh"

Mesh::Mesh(const Json::Value &json, const Transform* transform, const Material* material) {
    std::ifstream fin(json.asString());
    if (!fin.is_open()) {
        std::cerr << "Failed to open mesh file: " << json.asString() << std::endl;
        exit(1);
    }

    bool isFree = (material != nullptr);
    std::string line;
    std::vector<Vector2f> u;
    std::vector<Vertex> vertexArray;
    std::vector<unsigned int> indices;
    while (getline(fin, line)) {
        std::vector<std::string> s = split(line, ' ');
        if (s[0] == "v") {
            Vector3f x(std::stod(s[1]), std::stod(s[2]), std::stod(s[3]));
            x = transform->applyTo(x);
            vertexArray.emplace_back(x, isFree);
        } else if (s[0] == "vt")
            u.emplace_back(std::stof(s[1]), std::stof(s[2]));
        else if (s[0] == "f") {
            int index0, index1, index2, uIndex0, uIndex1, uIndex2;
            if (line.find('/') != std::string::npos) {
                std::vector<std::string> t;
                t = split(s[1], '/');
                index0 = std::stoi(t[0]) - 1;
                uIndex0 = std::stoi(t[1]) - 1;
                t = split(s[2], '/');
                index1 = std::stoi(t[0]) - 1;
                uIndex1 = std::stoi(t[1]) - 1;
                t = split(s[3], '/');
                index2 = std::stoi(t[0]) - 1;
                uIndex2 = std::stoi(t[1]) - 1;
                vertexArray[index0].u = u[uIndex0];
                vertexArray[index1].u = u[uIndex1];
                vertexArray[index2].u = u[uIndex2];
            } else {
                index0 = std::stoi(s[1]) - 1;
                index1 = std::stoi(s[2]) - 1;
                index2 = std::stoi(s[3]) - 1;
            }
            indices.push_back(index0);
            indices.push_back(index1);
            indices.push_back(index2);
        }
    }
    fin.close();

    initialize(vertexArray, indices, material);
}

Mesh::~Mesh() {
    for (const Vertex* vertex : vertices)
        delete vertex;
    for (const Edge* edge : edges)
        delete edge;
    for (const Face* face : faces)
        delete face;

    if (gpu) {
        deleteVertices<<<GRID_SIZE, BLOCK_SIZE>>>(verticesGpu.size(), thrust::raw_pointer_cast(verticesGpu.data()));
        CUDA_CHECK_LAST();
        deleteEdges<<<GRID_SIZE, BLOCK_SIZE>>>(edgesGpu.size(), thrust::raw_pointer_cast(edgesGpu.data()));
        CUDA_CHECK_LAST();
        deleteFaces<<<GRID_SIZE, BLOCK_SIZE>>>(facesGpu.size(), thrust::raw_pointer_cast(facesGpu.data()));
        CUDA_CHECK_LAST();
    }
}

std::vector<std::string> Mesh::split(const std::string& s, char c) const {
    std::string t = s;
    std::vector<std::string> ans;
    while (t.find(c) != std::string::npos) {
        unsigned int index = t.find(c);
        ans.push_back(t.substr(0, index));
        t.erase(0, index + 1);
    }
    ans.push_back(t);
    return ans;
}

Edge* Mesh::findEdge(int index0, int index1, std::map<std::pair<int, int>, int>& edgeMap) {
    if (index0 > index1)
        std::swap(index0, index1);
    std::pair<int, int> pair = std::make_pair(index0, index1);
    std::map<std::pair<int, int>, int>::const_iterator iter = edgeMap.find(pair);
    if (iter != edgeMap.end())
        return edges[iter->second];
    else {
        edgeMap[pair] = edges.size();
        edges.push_back(new Edge(vertices[index0], vertices[index1]));
        return edges.back();
    }
}

void Mesh::initialize(const std::vector<Vertex>& vertexArray, const std::vector<unsigned int>& indices, const Material* material) {
    if (!gpu) {
        vertices.resize(vertexArray.size());
        for (int i = 0; i < vertexArray.size(); i++) {
            vertices[i] = new Vertex(vertexArray[i].x, vertexArray[i].isFree);
            *vertices[i] = vertexArray[i];
        }

        std::map<std::pair<int, int>, int> edgeMap;
        for (int i = 0; i < indices.size(); i += 3) {
            int index0 = indices[i];
            int index1 = indices[i + 1];
            int index2 = indices[i + 2];

            Vertex* vertex0 = vertices[index0];
            Vertex* vertex1 = vertices[index1];
            Vertex* vertex2 = vertices[index2];
            Edge* edge0 = findEdge(index0, index1, edgeMap);
            Edge* edge1 = findEdge(index1, index2, edgeMap);
            Edge* edge2 = findEdge(index2, index0, edgeMap);
            Face* face = new Face(vertex0, vertex1, vertex2, material);

            edge0->setOppositeAndAdjacent(vertex2, face);
            edge1->setOppositeAndAdjacent(vertex0, face);
            edge2->setOppositeAndAdjacent(vertex1, face);
            face->setEdges(edge0, edge1, edge2);

            faces.push_back(face);
        }

        for (const Edge* edge : edges)
            if (edge->isBoundary())
                for (int i = 0; i < 2; i++)
                    edge->getVertex(i)->preserve = true;
    } else {
        int nVertices = vertexArray.size();
        int nFaces = indices.size() / 3;
        int nEdges = 3 * nFaces;
        thrust::device_vector<Vertex> vertexArrayGpu = vertexArray;
        thrust::device_vector<unsigned int> indicesGpu = indices;
        
        verticesGpu.resize(nVertices);
        Vertex** verticesPointer = thrust::raw_pointer_cast(verticesGpu.data());
        initializeVertices<<<GRID_SIZE, BLOCK_SIZE>>>(nVertices, thrust::raw_pointer_cast(vertexArrayGpu.data()), verticesPointer);
        CUDA_CHECK_LAST();

        facesGpu.resize(nFaces);
        Face** facesPointer = thrust::raw_pointer_cast(facesGpu.data());
        thrust::device_vector<thrust::pair<int, int>> edgeIndices(nEdges);
        thrust::pair<int, int>* edgeIndicesPointer = thrust::raw_pointer_cast(edgeIndices.data());
        thrust::device_vector<EdgeData> edgeData(nEdges);
        EdgeData* edgeDataPointer = thrust::raw_pointer_cast(edgeData.data());
        initializeFaces<<<GRID_SIZE, BLOCK_SIZE>>>(nFaces, thrust::raw_pointer_cast(indicesGpu.data()), verticesPointer, material, facesPointer, edgeIndicesPointer, edgeDataPointer);
        CUDA_CHECK_LAST();
        thrust::sort_by_key(edgeIndices.begin(), edgeIndices.end(), edgeData.begin());

        edgesGpu.resize(nEdges);
        Edge** edgesPointer = thrust::raw_pointer_cast(edgesGpu.data());
        initializeEdges<<<GRID_SIZE, BLOCK_SIZE>>>(nEdges, edgeIndicesPointer, edgeDataPointer, verticesPointer, edgesPointer);
        CUDA_CHECK_LAST();
        setupEdges<<<GRID_SIZE, BLOCK_SIZE>>>(nEdges, edgeIndicesPointer, edgeDataPointer, edgesPointer);
        CUDA_CHECK_LAST();
        edgesGpu.erase(thrust::remove_if(edgesGpu.begin(), edgesGpu.end(), IsNull()), edgesGpu.end());
    }

    updateIndices();
    updateGeometries();
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

void Mesh::readDataFromFile(const std::string& path) {
    std::ifstream fin("input.txt");
    for (Vertex* vertex : vertices)
        fin >> vertex->x0(0) >> vertex->x0(1) >> vertex->x0(2) >> vertex->x(0) >> vertex->x(1) >> vertex->x(2) >> vertex->v(0) >> vertex->v(1) >> vertex->v(2);
    fin.close();
}

void Mesh::apply(const Operator& op) {
    for (const Vertex* vertex : op.removedVertices)
        vertices.erase(std::remove(vertices.begin(), vertices.end(), vertex), vertices.end());
    vertices.insert(vertices.end(), op.addedVertices.begin(), op.addedVertices.end());
    
    for (const Edge* edge : op.removedEdges)
        edges.erase(std::remove(edges.begin(), edges.end(), edge), edges.end());
    edges.insert(edges.end(), op.addedEdges.begin(), op.addedEdges.end());

    for (const Face* face : op.removedFaces)
        faces.erase(std::remove(faces.begin(), faces.end(), face), faces.end());
    faces.insert(faces.end(), op.addedFaces.begin(), op.addedFaces.end());

    for (const Vertex* vertex : op.removedVertices)
        delete vertex;
    for (const Edge* edge : op.removedEdges)
        delete edge;
    for (const Face* face : op.removedFaces)
        delete face;
}

void Mesh::reset() {
    for (Vertex* vertex : vertices)
        vertex->x = vertex->x0;
}

void Mesh::updateGeometries() {
    if (!gpu) {
        for (Face* face : faces)
            face->update();
        for (Edge* edge : edges)
            edge->update();
        for (Vertex* vertex : vertices) {
            vertex->x1 = vertex->x;
            vertex->m = 0.0f;
            vertex->n = Vector3f();
        }
        for (const Face* face : faces) {
            float m = face->getMass() / 3.0f;
            for (int i = 0; i < 3; i++) {
                Vertex* vertex = face->getVertex(i);
                Vector3f e0 = face->getVertex((i + 1) % 3)->x - vertex->x;
                Vector3f e1 = face->getVertex((i + 2) % 3)->x - vertex->x;
                face->getVertex(i)->m += m;
                vertex->n += e0.cross(e1) / (e0.norm2() * e1.norm2());
            }
        }
        for (Vertex* vertex : vertices)
            vertex->n.normalize();
    } else {
        thrust::device_vector<int> indices(3 * facesGpu.size());
        thrust::device_vector<VertexData> vertexData(3 * facesGpu.size());

        updateGeometriesFaces<<<GRID_SIZE, BLOCK_SIZE>>>(facesGpu.size(), thrust::raw_pointer_cast(facesGpu.data()), thrust::raw_pointer_cast(indices.data()), thrust::raw_pointer_cast(vertexData.data()));
        CUDA_CHECK_LAST();
        updateGeometriesEdges<<<GRID_SIZE, BLOCK_SIZE>>>(edgesGpu.size(), thrust::raw_pointer_cast(edgesGpu.data()));
        CUDA_CHECK_LAST();
        
        thrust::sort_by_key(indices.begin(), indices.end(), vertexData.begin());
        thrust::device_vector<int> outputIndices(3 * facesGpu.size());
        thrust::device_vector<VertexData> outputVertexData(3 * facesGpu.size());
        thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<VertexData>::iterator> iter = thrust::reduce_by_key(indices.begin(), indices.end(), vertexData.begin(), outputIndices.begin(), outputVertexData.begin());
        updateGeometriesVertices<<<GRID_SIZE, BLOCK_SIZE>>>(iter.first - outputIndices.begin(), thrust::raw_pointer_cast(outputIndices.data()), thrust::raw_pointer_cast(outputVertexData.data()), thrust::raw_pointer_cast(verticesGpu.data()));
        CUDA_CHECK_LAST();
    }
}

void Mesh::updateIndices() {
    if (!gpu)
        for (int i = 0; i < vertices.size(); i++)
            vertices[i]->index = i;
    else {
        updateIndicesGpu<<<GRID_SIZE, BLOCK_SIZE>>>(verticesGpu.size(), thrust::raw_pointer_cast(verticesGpu.data()));
        CUDA_CHECK_LAST();
    }
}

void Mesh::updateVelocities(float dt) {
    float invDt = 1.0f / dt;
    for (Vertex* vertex : vertices)
        vertex->v = (vertex->x - vertex->x0) * invDt;
}

void Mesh::updateRenderingData(bool rebind) {
    if (!gpu) {
        std::vector<Vertex> vertexArray(vertices.size(), Vertex(Vector3f(), true));
        for (int i = 0; i < vertices.size(); i++)
            vertexArray[i] = *vertices[i];

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vertexArray.size() * sizeof(Vertex), vertexArray.data(), GL_DYNAMIC_DRAW);  

        if (rebind) {
            std::vector<unsigned int> edgeIndices(2 * edges.size());
            for (int i = 0; i < edges.size(); i++)
                for (int j = 0; j < 2; j++)
                    edgeIndices[2 * i + j] = edges[i]->getVertex(j)->index;

            std::vector<unsigned int> faceIndices(3 * faces.size());
            for (int i = 0; i < faces.size(); i++)
                for (int j = 0; j < 3; j++)
                    faceIndices[3 * i + j] = faces[i]->getVertex(j)->index;

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgeEbo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, edgeIndices.size() * sizeof(unsigned int), edgeIndices.data(), GL_DYNAMIC_DRAW);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, faceEbo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, faceIndices.size() * sizeof(unsigned int), faceIndices.data(), GL_DYNAMIC_DRAW);
        }
    } else if (!rebind) {
        CUDA_CHECK(cudaGraphicsMapResources(1, &vboCuda));
        Vertex* vertexArray;
        size_t sizeVertexArray;
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&vertexArray), &sizeVertexArray, vboCuda));
        updateRenderingDataVertices<<<GRID_SIZE, BLOCK_SIZE>>>(verticesGpu.size(), thrust::raw_pointer_cast(verticesGpu.data()), vertexArray);
        CUDA_CHECK_LAST();

        CUDA_CHECK(cudaGraphicsUnmapResources(1, &vboCuda));
    } else {
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, verticesGpu.size() * sizeof(Vertex), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgeEbo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 2 * edgesGpu.size() * sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, faceEbo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * facesGpu.size() * sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);

        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&vboCuda, vbo, cudaGraphicsRegisterFlagsWriteDiscard));
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&edgeEboCuda, edgeEbo, cudaGraphicsRegisterFlagsWriteDiscard));
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&faceEboCuda, faceEbo, cudaGraphicsRegisterFlagsWriteDiscard));

        CUDA_CHECK(cudaGraphicsMapResources(1, &vboCuda));
        Vertex* vertexArray;
        size_t sizeVertexArray;
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&vertexArray), &sizeVertexArray, vboCuda));
        updateRenderingDataVertices<<<GRID_SIZE, BLOCK_SIZE>>>(verticesGpu.size(), thrust::raw_pointer_cast(verticesGpu.data()), vertexArray);
        CUDA_CHECK_LAST();

        CUDA_CHECK(cudaGraphicsMapResources(1, &edgeEboCuda));
        unsigned int* edgeIndices;
        size_t sizeEdgeIndices;
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&edgeIndices), &sizeEdgeIndices, edgeEboCuda));
        updateRenderingDataEdges<<<GRID_SIZE, BLOCK_SIZE>>>(edgesGpu.size(), thrust::raw_pointer_cast(edgesGpu.data()), edgeIndices);
        CUDA_CHECK_LAST();

        CUDA_CHECK(cudaGraphicsMapResources(1, &faceEboCuda));
        unsigned int* faceIndices;
        size_t sizeFaceIndices;
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&faceIndices), &sizeFaceIndices, faceEboCuda));
        updateRenderingDataFaces<<<GRID_SIZE, BLOCK_SIZE>>>(facesGpu.size(), thrust::raw_pointer_cast(facesGpu.data()), faceIndices);
        CUDA_CHECK_LAST();

        CUDA_CHECK(cudaGraphicsUnmapResources(1, &vboCuda));
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &edgeEboCuda));
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &faceEboCuda));
    }
}

void Mesh::bind(const Material* material) {
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glGenVertexArrays(1, &edgeVao);
    glGenBuffers(1, &edgeEbo);
    glBindVertexArray(edgeVao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgeEbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, x)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, n)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, u)));

    glGenVertexArrays(1, &faceVao);
    glGenBuffers(1, &faceEbo);
    glBindVertexArray(faceVao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, faceEbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, x)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, n)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, u)));

    updateRenderingData(true);
}

void Mesh::renderEdges() const {
    glBindVertexArray(edgeVao);
    glDrawElements(GL_LINES, 2 * (!gpu ? edges.size() : edgesGpu.size()), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void Mesh::renderFaces() const {
    glBindVertexArray(faceVao);
    glDrawElements(GL_TRIANGLES, 3 * (!gpu ? faces.size() : facesGpu.size()), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}
