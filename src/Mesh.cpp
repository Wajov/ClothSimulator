#include "Mesh.hpp"

Mesh::Mesh(const Json::Value &json, const Transform* transform, const Material* material) {
    std::ifstream fin(json.asString());
    if (!fin.is_open()) {
        std::cerr << "Failed to open mesh file: " << json.asString() << std::endl;
        exit(1);
    }

    bool isFree = (material != nullptr);
    std::string line;
    std::vector<Vector2f> u;
    std::vector<int> vertexMap;
    std::map<std::pair<int, int>, int> edgeMap;
    while (getline(fin, line)) {
        std::vector<std::string> s = split(line, ' ');
        if (s[0] == "v") {
            Vector3f x(std::stod(s[1]), std::stod(s[2]), std::stod(s[3]));
            x = transform->applyTo(x);
            vertices.push_back(new Vertex(vertices.size(), x, isFree));
            vertexMap.push_back(-1);
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
                vertices[index0]->u = u[uIndex0];
                vertices[index1]->u = u[uIndex1];
                vertices[index2]->u = u[uIndex2];
                vertexMap[index0] = uIndex0;
                vertexMap[index1] = uIndex1;
                vertexMap[index2] = uIndex2;
            } else {
                index0 = std::stoi(s[1]) - 1;
                index1 = std::stoi(s[2]) - 1;
                index2 = std::stoi(s[3]) - 1;
            }

            Vertex* vertex0 = vertices[index0];
            Vertex* vertex1 = vertices[index1];
            Vertex* vertex2 = vertices[index2];
            Edge* edge0 = findEdge(vertex0, vertex1, edgeMap);
            Edge* edge1 = findEdge(vertex1, vertex2, edgeMap);
            Edge* edge2 = findEdge(vertex2, vertex0, edgeMap);
            Face* face = new Face(vertex0, vertex1, vertex2, material);

            edge0->setOppositeAndAdjacent(vertex2, face);
            edge1->setOppositeAndAdjacent(vertex0, face);
            edge2->setOppositeAndAdjacent(vertex1, face);
            face->setEdges(edge0, edge1, edge2);

            faces.push_back(face);
        }
    }
    fin.close();

    for (const Edge* edge : edges)
        if (edge->isBoundary())
            for (int i = 0; i < 2; i++)
                edge->getVertex(i)->preserve = true;

    updateGeometries();
}

Mesh::~Mesh() {
    for (const Edge* edge : edges)
        delete edge;
    for (const Face* face : faces)
        delete face;
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

Edge* Mesh::findEdge(const Vertex* vertex0, const Vertex* vertex1, std::map<std::pair<int, int>, int>& edgeMap) {
    int index0 = vertex0->index;
    int index1 = vertex1->index;
    std::pair<int, int> pair = index0 < index1 ? std::make_pair(index0, index1) : std::make_pair(index1, index0);
    std::map<std::pair<int, int>, int>::const_iterator iter = edgeMap.find(pair);
    if (iter != edgeMap.end())
        return edges[iter->second];
    else {
        edgeMap[pair] = edges.size();
        edges.push_back(new Edge(vertex0, vertex1));
        return edges.back();
    }
}

std::vector<Vertex*>& Mesh::getVertices() {
    return vertices;
}

std::vector<Edge*>& Mesh::getEdges() {
    return edges;
}

std::vector<Face*>& Mesh::getFaces() {
    return faces;
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
}

void Mesh::updateVelocities(float dt) {
    float invDt = 1.0f / dt;
    for (Vertex* vertex : vertices)
        vertex->v = (vertex->x - vertex->x0) * invDt;
}

void Mesh::updateIndices() {
    for (int i = 0; i < vertices.size(); i++)
        vertices[i]->index = i;
}

void Mesh::updateRenderingData(bool rebind) {
    vertexArray.resize(vertices.size(), Vertex(0, Vector3f(), true));
    for (int i = 0; i < vertices.size(); i++)
        vertexArray[i] = *vertices[i];

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertexArray.size() * sizeof(Vertex), vertexArray.data(), GL_DYNAMIC_DRAW);  

    if (rebind) {
        edgeIndices.resize(2 * edges.size());
        for (int i = 0; i < edges.size(); i++)
            for (int j = 0; j < 2; j++)
                edgeIndices[2 * i + j] = edges[i]->getVertex(j)->index;

        faceIndices.resize(3 * faces.size());
        for (int i = 0; i < faces.size(); i++)
            for (int j = 0; j < 3; j++)
                faceIndices[3 * i + j] = faces[i]->getVertex(j)->index;
                
        glBindVertexArray(edgeVao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgeEbo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, edgeIndices.size() * sizeof(unsigned int), edgeIndices.data(), GL_DYNAMIC_DRAW);

        glBindVertexArray(faceVao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, faceEbo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, faceIndices.size() * sizeof(unsigned int), faceIndices.data(), GL_DYNAMIC_DRAW);
    }
}

void Mesh::bind() {
        vertexArray.resize(vertices.size(), Vertex(0, Vector3f(), true));
    for (int i = 0; i < vertices.size(); i++)
        vertexArray[i] = *vertices[i];

    edgeIndices.resize(2 * edges.size());
    for (int i = 0; i < edges.size(); i++)
        for (int j = 0; j < 2; j++)
            edgeIndices[2 * i + j] = edges[i]->getVertex(j)->index;

    faceIndices.resize(3 * faces.size());
    for (int i = 0; i < faces.size(); i++)
        for (int j = 0; j < 3; j++)
            faceIndices[3 * i + j] = faces[i]->getVertex(j)->index;

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertexArray.size() * sizeof(Vertex), vertexArray.data(), GL_DYNAMIC_DRAW);

    glGenVertexArrays(1, &edgeVao);
    glGenBuffers(1, &edgeEbo);
    glBindVertexArray(edgeVao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgeEbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, edgeIndices.size() * sizeof(unsigned int), edgeIndices.data(), GL_DYNAMIC_DRAW);
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
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faceIndices.size() * sizeof(unsigned int), faceIndices.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, x)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, n)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, u)));
}

void Mesh::renderEdge() const {
    glBindVertexArray(edgeVao);
    glDrawElements(GL_LINES, edgeIndices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void Mesh::renderFace() const {
    glBindVertexArray(faceVao);
    glDrawElements(GL_TRIANGLES, faceIndices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}
