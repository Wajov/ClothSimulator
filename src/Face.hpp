#ifndef FACE_HPP
#define FACE_HPP

#include "TypeHelper.hpp"
#include "MathHelper.hpp"
#include "Bounds.hpp"
#include "Vertex.hpp"
#include "Edge.hpp"
#include "Material.hpp"
#include "Remeshing.hpp"

class Edge;

class Face {
private:
    std::vector<Vertex*> vertices;
    std::vector<Edge*> edges;
    Vector3f normal;
    Matrix2x2f inverse;
    float area, mass;

public:
    Face(const Vertex* vertex0, const Vertex* vertex1, const Vertex* vertex2, const Material* material);
    ~Face();
    void initialize(const Material* material);
    Vertex* getVertex(int index) const;
    void replaceVertex(const Vertex* v, const Vertex* vertex);
    Edge* getEdge(int index) const;
    void setEdges(const Edge* edge0, const Edge* edge1, const Edge* edge2);
    void replaceEdge(const Edge* e, const Edge* edge);
    Vector3f getNormal() const;
    Matrix2x2f getInverse() const;
    float getArea() const;
    float getMass() const;
    bool contain(const Edge* edge) const;
    int sequence(const Edge* edge) const;
    Edge* findEdge(const Vertex* vertex0, const Vertex* vertex1) const;
    Edge* findOpposite(const Vertex* vertex) const;
    Bounds bounds(bool ccd) const;
    Matrix3x2f derivative(const Vector3f& v0, const Vector3f& v1, const Vector3f& v2) const;
    Matrix2x2f curvature() const;
    void update();
};

#endif