#ifndef MESH_HPP
#define MESH_HPP

#include <vector>
#include <map>
#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <json/json.h>

#include "TypeHelper.hpp"
#include "Vertex.hpp"
#include "Edge.hpp"
#include "Face.hpp"
#include "Transform.hpp"

class Mesh {
private:
    std::vector<Edge*> edges;
    std::vector<Face*> faces;
    std::vector<Vertex> vertices;
    std::vector<unsigned int> edgeIndices, faceIndices;
    unsigned int vbo, edgeVao, faceVao;
    std::vector<std::string> split(const std::string& s, char c) const;
    Edge* getEdge(const Vertex* vertex0, const Vertex* vertex1, std::map<std::pair<int, int>, int>& edgeMap) const;

public:
    Mesh(const Json::Value& json, const Transform* transform);
    ~Mesh();
    const std::vector<Vertex>& getVertices() const;
    const std::vector<Edge*>& getEdges() const;
    const std::vector<Face*>& getFaces() const;
    void readDataFromFile(const std::string& path);
    void renderEdge() const;
    void renderFace() const;
    void updateData(const Material* material);
    void update(float dt, const VectorXf& dv);
};

#endif