#include "Edge.hpp"

Edge::Edge(const Vertex* vertex0, const Vertex* vertex1) :
    vertices{const_cast<Vertex*>(vertex0), const_cast<Vertex*>(vertex1)} {}

Edge::~Edge() {}

Vertex* Edge::getVertex(int index) const {
    return vertices[index];
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

Bounds Edge::bounds(bool ccd) const {
    Bounds ans;
    for (int i = 0; i < 2; i++) {
        const Vertex* vertex = vertices[i];
        ans += vertex->x;
        if (ccd)
            ans += vertex->x0;
    }
    return ans;
}

void Edge::updateData() {
    length = (vertices[1]->x - vertices[0]->x).norm();
    if (adjacents.size() == 2) {
        Vector3f e = (vertices[0]->x - vertices[1]->x).normalized();
        Vector3f n0 = adjacents[0]->getNormal();
        Vector3f n1 = adjacents[1]->getNormal();
        float sine = e.dot(n0.cross(n1));
        float cosine = n0.dot(n1);
        angle = std::atan2(sine, cosine);
    } else
        angle = 0.0f;
}
