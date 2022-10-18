#include "Mesh.hpp"

Mesh::Mesh(const Json::Value &json, const Transform* transform, const Material* material) {
    std::ifstream fin(json.asString());
    if (!fin.is_open()) {
        std::cerr << "Failed to open mesh file: " << json.asString() << std::endl;
        exit(1);
    }

    bool isFree = (material != nullptr);
    std::string line;
    std::vector<Vector3f> u;
    std::vector<int> vertexMap;
    std::map<std::pair<int, int>, int> edgeMap;
    while (getline(fin, line)) {
        std::vector<std::string> s = split(line, ' ');
        if (s[0] == "v") {
            Vector3f x(std::stod(s[1]), std::stod(s[2]), std::stod(s[3]));
            x = transform->applyTo(x);
            vertices.emplace_back(vertices.size(), x, isFree);
            vertexMap.push_back(-1);
        } else if (s[0] == "vt") {
            if (s.size() == 4)
                u.push_back(Vector3f(std::stof(s[1]), std::stof(s[2]), std::stof(s[3])));
            else if (s.size() == 3)
                u.push_back(Vector3f(std::stof(s[1]), std::stof(s[2]), 0.0f));
        } else if (s[0] == "f") {
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
                assert(vertexMap[index0] == -1 || vertexMap[index0] == uIndex0);
                assert(vertexMap[index1] == -1 || vertexMap[index1] == uIndex1);
                assert(vertexMap[index2] == -1 || vertexMap[index2] == uIndex2);
                vertices[index0].u = u[uIndex0];
                vertices[index1].u = u[uIndex1];
                vertices[index2].u = u[uIndex2];
                vertexMap[index0] = uIndex0;
                vertexMap[index1] = uIndex1;
                vertexMap[index2] = uIndex2;
            } else {
                index0 = std::stoi(s[1]) - 1;
                index1 = std::stoi(s[2]) - 1;
                index2 = std::stoi(s[3]) - 1;
            }

            Vertex* vertex0 = &vertices[index0];
            Vertex* vertex1 = &vertices[index1];
            Vertex* vertex2 = &vertices[index2];
            Edge* edge0 = getEdge(vertex0, vertex1, edgeMap);
            Edge* edge1 = getEdge(vertex1, vertex2, edgeMap);
            Edge* edge2 = getEdge(vertex2, vertex0, edgeMap);
            Face* face = new Face(vertex0, vertex1, vertex2);

            edge0->addOpposite(vertex2);
            edge0->addAdjacent(face);
            edge1->addOpposite(vertex0);
            edge1->addAdjacent(face);
            edge2->addOpposite(vertex1);
            edge2->addAdjacent(face);
            face->setEdges(edge0, edge1, edge2);

            faces.push_back(face);

            edgeIndices.push_back(index0);
            edgeIndices.push_back(index1);
            edgeIndices.push_back(index1);
            edgeIndices.push_back(index2);
            edgeIndices.push_back(index2);
            edgeIndices.push_back(index0);

            faceIndices.push_back(index0);
            faceIndices.push_back(index1);
            faceIndices.push_back(index2);
        }
    }
    fin.close();

    update(material);

    unsigned int edgeEbo, faceEbo;

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_DYNAMIC_DRAW);

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
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, u)));

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
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, u)));
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

Edge* Mesh::getEdge(const Vertex* vertex0, const Vertex* vertex1, std::map<std::pair<int, int>, int>& edgeMap) const {
    int index0 = vertex0->index;
    int index1 = vertex1->index;
    std::pair<int, int> pair = index0 < index1 ? std::make_pair(index0, index1) : std::make_pair(index1, index0);
    std::map<std::pair<int, int>, int>::const_iterator iter = edgeMap.find(pair);
    return iter != edgeMap.end() ? edges[iter->second] : new Edge(vertex0, vertex1);
}

std::vector<Vertex>& Mesh::getVertices() {
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
    for (Vertex& vertex : vertices)
        fin >> vertex.x0(0) >> vertex.x0(1) >> vertex.x0(2) >> vertex.x(0) >> vertex.x(1) >> vertex.x(2) >> vertex.v(0) >> vertex.v(1) >> vertex.v(2);
    fin.close();
}

void Mesh::update(const Material* material) {
    for (Face* face : faces)
        face->update(material);
    for (Edge* edge : edges)
        edge->update();
    for (Vertex& vertex : vertices) {
        vertex.m = 0.0f;
        vertex.n = Vector3f(0.0f, 0.0f, 0.0f);
    }
    for (const Face* face : faces) {
        float m = face->getMass() / 3.0f;
        Vector3f n = face->getArea() * face->getNormal();
        for (int i = 0; i < 3; i++) {
            face->getVertex(i)->m += m;
            face->getVertex(i)->n += n;
        }
    }
    for (Vertex& vertex : vertices)
        vertex.n.normalized();
}

void Mesh::updateRenderingData() const {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_DYNAMIC_DRAW);
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
