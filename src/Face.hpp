#ifndef FACE_HPP
#define FACE_HPP

#include "TypeHelper.hpp"
#include "MathHelper.hpp"
#include "Bounds.hpp"
#include "Vertex.hpp"
#include "Edge.hpp"
#include "Material.hpp"

class Edge;

class Face {
private:
    std::vector<Vertex*> vertices;
    std::vector<Edge*> edges;
    Vector3f normal;
    Matrix3x3f inverse;
    float area, mass;

public:
    Face(const Vertex* vertex0, const Vertex* vertex1, const Vertex* vertex2);
    ~Face();
    Vertex* getVertex(int index) const;
    Edge* getEdge(int index) const;
    void setEdges(const Edge* edge0, const Edge* edge1, const Edge* edge2);
    Vector3f getNormal() const;
    Matrix3x3f getInverse() const;
    float getArea() const;
    float getMass() const;
    Bounds bounds(bool ccd) const;
    void update(const Material* material);
};

#endif