#ifndef MESH_HPP
#define MESH_HPP

#include <vector>
#include <map>
#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <json/json.h>

#include "Vector.hpp"
#include "Vertex.hpp"
#include "Edge.hpp"
#include "Face.hpp"
#include "Transform.hpp"
#include "Operator.hpp"

class Mesh {
private:
    std::vector<Vertex*> vertices;
    std::vector<Edge*> edges;
    std::vector<Face*> faces;
    std::vector<Vertex> vertexArray;
    std::vector<unsigned int> edgeIndices, faceIndices;
    unsigned int vbo, edgeVao, edgeEbo, faceVao, faceEbo;
    std::vector<std::string> split(const std::string& s, char c) const;
    Edge* findEdge(const Vertex* vertex0, const Vertex* vertex1, std::map<std::pair<int, int>, int>& edgeMap);

public:
    Mesh(const Json::Value& json, const Transform* transform, const Material* material);
    ~Mesh();
    std::vector<Vertex*>& getVertices();
    std::vector<Edge*>& getEdges();
    std::vector<Face*>& getFaces();
    void readDataFromFile(const std::string& path);
    void apply(const Operator& op);
    void reset();
    void updateGeometries();
    void updateVelocities(float dt);
    void updateIndices();
    void updateRenderingData(bool rebind);
    void bind();
    void renderEdge() const;
    void renderFace() const;
};

#endif