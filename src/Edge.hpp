#ifndef EDGE_HPP
#define EDGE_HPP

#include <vector>

#include "Vertex.hpp"
#include "Face.hpp"

class Edge {
private:
    Vertex* v0, * v1;
    std::vector<Vertex*> opposites;
    std::vector<Face*> adjacents;
    float length, angle;

public:
    Edge(const Vertex* v0, const Vertex* v1);
    ~Edge();
    Vertex* getV0() const;
    Vertex* getV1() const;
    void addOpposite(const Vertex* vertex);
    const std::vector<Vertex*>& getOpposites() const;
    void addAdjacent(const Face* face);
    const std::vector<Face*>& getAdjacents() const;
    float getLength() const;
    float getAngle() const;
    void updateData();
};

#endif