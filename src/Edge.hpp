#ifndef EDGE_HPP
#define EDGE_HPP

#include <vector>

#include "Bounds.hpp"
#include "Vertex.hpp"
#include "Face.hpp"

class Face;

class Edge {
private:
    std::vector<Vertex*> vertices, opposites;
    std::vector<Face*> adjacents;
    float length, angle;

public:
    Edge(const Vertex* vertex0, const Vertex* vertex1);
    ~Edge();
    Vertex* getVertex(int index) const;
    void addOpposite(const Vertex* vertex);
    const std::vector<Vertex*>& getOpposites() const;
    void addAdjacent(const Face* face);
    const std::vector<Face*>& getAdjacents() const;
    float getLength() const;
    float getAngle() const;
    Bounds bounds(bool ccd) const;
    void updateData();
};

#endif