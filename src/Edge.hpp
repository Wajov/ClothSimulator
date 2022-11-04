#ifndef EDGE_HPP
#define EDGE_HPP

#include <vector>

#include "Bounds.hpp"
#include "Vertex.hpp"
#include "Face.hpp"

class Face;

class Edge {
private:
    Vertex* vertices[2], * opposites[2];
    Face* adjacents[2];
    float length, angle;

public:
    Edge(const Vertex* vertex0, const Vertex* vertex1);
    ~Edge();
    Vertex* getVertex(int index) const;
    void replaceVertex(const Vertex* v, const Vertex* vertex);
    Vertex* getOpposite(int index) const;
    void replaceOpposite(const Vertex* v, const Vertex* vertex);
    Face* getAdjacent(int index) const;
    void replaceAdjacent(const Face* f, const Face* face);
    void setOppositeAndAdjacent(const Vertex* vertex, const Face* face);
    float getLength() const;
    float getAngle() const;
    bool contain(const Vertex* vertex) const;
    bool isBoundary() const;
    Bounds bounds(bool ccd) const;
    void update();
};

#endif