#include "Edge.cuh"

Edge::Edge(const Vertex* vertex0, const Vertex* vertex1) :
    index(0),
    vertices{const_cast<Vertex*>(vertex0), const_cast<Vertex*>(vertex1)},
    opposites{nullptr, nullptr},
    adjacents{nullptr, nullptr} {}

Edge::~Edge() {}

int Edge::getIndex() const {
    return index;
}

void Edge::setIndex(int index) {
    this->index = index;
}

Vertex* Edge::getVertex(int index) const {
    return vertices[index];
}

void Edge::replaceVertex(const Vertex* v, const Vertex* vertex) {
    for (int i = 0; i < 2; i++)
        if (vertices[i] == v) {
            vertices[i] = const_cast<Vertex*>(vertex);
            return;
        }
}

Vertex* Edge::getOpposite(int index) const {
    return opposites[index];
}

void Edge::replaceOpposite(const Vertex* v, const Vertex* vertex) {
    for (int i = 0; i < 2; i++)
        if (opposites[i] == v) {
            opposites[i] = const_cast<Vertex*>(vertex);
            return;
        }
}

Face* Edge::getAdjacent(int index) const {
    return adjacents[index];
}

void Edge::replaceAdjacent(const Face* f, const Face* face) {
    for (int i = 0; i < 2; i++)
        if (adjacents[i] == f) {
            adjacents[i] = const_cast<Face*>(face);
            return;
        }
}

void Edge::setOppositeAndAdjacent(const Vertex* vertex, const Face* face) {
    int i = face->sequence(this);
    opposites[i] = const_cast<Vertex*>(vertex);
    adjacents[i] = const_cast<Face*>(face);
}

float Edge::getLength() const {
    return length;
}

float Edge::getAngle() const {
    return angle;
}

bool Edge::contain(const Vertex* vertex) const {
    for (int i = 0; i < 2; i++)
        if (vertices[i] == vertex)
            return true;
    return false;
}

bool Edge::isBoundary() const {
    return opposites[0] == nullptr || opposites[1] == nullptr;
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

void Edge::update() {
    length = (vertices[1]->x - vertices[0]->x).norm();
    if (!isBoundary()) {
        Vector3f e = (vertices[0]->x - vertices[1]->x).normalized();
        Vector3f n0 = adjacents[0]->getNormal();
        Vector3f n1 = adjacents[1]->getNormal();
        float sine = e.dot(n0.cross(n1));
        float cosine = n0.dot(n1);
        angle = std::atan2(sine, cosine);
    } else
        angle = 0.0f;
}
