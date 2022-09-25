#include "Mesh.hpp"

Mesh::Mesh(const Json::Value &json, const Vector3f& translate) {
    std::ifstream fin(json.asString());
    if (!fin.is_open()) {
        std::cerr << "Failed to open mesh file: " << json.asString() << std::endl;
        exit(1);
    }

    std::string line;
    std::vector<Vector3f> uvs;
    std::vector<int> vertexMap;
    std::map<std::pair<int, int>, int> edgeMap;
    while (getline(fin, line)) {
        std::vector<std::string> s = split(line, ' ');
        if (s[0] == "v") {
            Vector3f position(std::stod(s[1]), std::stod(s[2]), std::stod(s[3]));
            position += translate;
            vertices.emplace_back(vertices.size(), position);
            vertexMap.push_back(-1);
        } else if (s[0] == "vt") {
            if (s.size() == 4)
                uvs.push_back(Vector3f(std::stof(s[1]), std::stof(s[2]), std::stof(s[3])));
            else if (s.size() == 3)
                uvs.push_back(Vector3f(std::stof(s[1]), std::stof(s[2]), 0.0f));
        } else if (s[0] == "f") {
            int index0, index1, index2, uvIndex0, uvIndex1, uvIndex2;
            if (line.find('/') != std::string::npos) {
                std::vector<std::string> t;
                t = split(s[1], '/');
                index0 = std::stoi(t[0]) - 1;
                uvIndex0 = std::stoi(t[1]) - 1;
                t = split(s[2], '/');
                index1 = std::stoi(t[0]) - 1;
                uvIndex1 = std::stoi(t[1]) - 1;
                t = split(s[3], '/');
                index2 = std::stoi(t[0]) - 1;
                uvIndex2 = std::stoi(t[1]) - 1;
                assert(vertexMap[index0] == -1 || vertexMap[index0] == uvIndex0);
                assert(vertexMap[index1] == -1 || vertexMap[index1] == uvIndex1);
                assert(vertexMap[index2] == -1 || vertexMap[index2] == uvIndex2);
                vertices[index0].uv = uvs[uvIndex0];
                vertices[index1].uv = uvs[uvIndex1];
                vertices[index2].uv = uvs[uvIndex2];
                vertexMap[index0] = uvIndex0;
                vertexMap[index1] = uvIndex1;
                vertexMap[index2] = uvIndex2;
            } else {
                index0 = std::stoi(s[1]) - 1;
                index1 = std::stoi(s[2]) - 1;
                index2 = std::stoi(s[3]) - 1;
            }

            Vertex* v0 = &vertices[index0];
            Vertex* v1 = &vertices[index1];
            Vertex* v2 = &vertices[index2];
            Face* face = new Face(v0, v1, v2);

            addEdge(index0, index1, v2, face, edgeMap);
            addEdge(index1, index2, v0, face, edgeMap);
            addEdge(index2, index0, v1, face, edgeMap);

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
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, position));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, uv));

    glGenVertexArrays(1, &faceVao);
    glGenBuffers(1, &faceEbo);
    glBindVertexArray(faceVao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, faceEbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faceIndices.size() * sizeof(unsigned int), faceIndices.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, position));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, uv));
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

void Mesh::addEdge(int index0, int index1, const Vertex* opposite, const Face* adjacent, std::map<std::pair<int, int>, int>& map) {
    std::pair<int, int> pair = index0 < index1 ? std::make_pair(index0, index1) : std::make_pair(index1, index0);
    std::map<std::pair<int, int>, int>::const_iterator iter = map.find(pair);
    if (iter != map.end()) {
        edges[iter->second]->addOpposite(opposite);
        edges[iter->second]->addAdjacent(adjacent);
    } else {
        map[pair] = edges.size();
        Edge* edge = new Edge(&vertices[index0], &vertices[index1]);
        edge->addOpposite(opposite);
        edge->addAdjacent(adjacent);
        edges.push_back(edge);
    }
}

const std::vector<Vertex>& Mesh::getVertices() const {
    return vertices;
}

const std::vector<Edge*>& Mesh::getEdges() const {
    return edges;
}

const std::vector<Face*>& Mesh::getFaces() const {
    return faces;
}

void Mesh::readDataFromFile(const std::string& path) {
    std::ifstream fin("input.txt");
    for (Vertex& vertex : vertices)
        fin >> vertex.position.x() >> vertex.position.y() >> vertex.position.z() >> vertex.velocity.x() >> vertex.velocity.y() >> vertex.velocity.z();
    fin.close();
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

void Mesh::updateData(const Material* material) {
    for (Face* face : faces)
        face->updateData(material);
    for (Edge* edge : edges)
        edge->updateData();
    for (Vertex& vertex : vertices) {
        vertex.mass = 0.0f;
        vertex.normal = Vector3f(0.0f, 0.0f, 0.0f);
    }
    for (const Face* face : faces) {
        float mass = face->getMass() / 3.0f;
        Vector3f normal = face->getArea() * face->getNormal();
        face->getV0()->mass += mass;
        face->getV1()->mass += mass;
        face->getV2()->mass += mass;
        face->getV0()->normal += normal;
        face->getV1()->normal += normal;
        face->getV2()->normal += normal;
    }
    for (Vertex& vertex : vertices)
        vertex.normal.normalized();
}

void Mesh::update(float dt, const VectorXf& dv) {
    for (int i = 0; i < vertices.size(); i++) {
        vertices[i].velocity += dv.block<3, 1>(3 * i, 0);
        vertices[i].position += vertices[i].velocity * dt;
    }

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_DYNAMIC_DRAW);
}