#include "Edge.hpp"

Edge::Edge(const Vertex* v0, const Vertex* v1) :
    v0(const_cast<Vertex*>(v0)),
    v1(const_cast<Vertex*>(v1)) {}

Edge::~Edge() {}

Vertex* Edge::getV0() const {
    return v0;
}

Vertex* Edge::getV1() const {
    return v1;
}

void Edge::addOpposite(const Vertex* vertex) {
    opposites.push_back(const_cast<Vertex*>(vertex));
    assert(opposites.size() < 3);
}

const std::vector<Vertex*>& Edge::getOpposites() const {
    return opposites;
}

void Edge::addAdjacent(const Face* face) {
    adjacents.push_back(const_cast<Face*>(face));
    assert(adjacents.size() < 3);
}

const std::vector<Face*>& Edge::getAdjacents() const {
    return adjacents;
}

float Edge::getLength() const {
    return length;
}

float Edge::getAngle() const {
    return angle;
}

void Edge::updateData() {
    length = (v1->position - v0->position).norm();
    if (adjacents.size() == 2) {
        Vector3f e = (v0->position - v1->position).normalized();
        Vector3f n0 = adjacents[0]->getNormal();
        Vector3f n1 = adjacents[1]->getNormal();
        float sine = e.dot(n0.cross(n1));
        float cosine = n0.dot(n1);
        angle = std::atan2(sine, cosine);
    } else
        angle = 0.0f;
}