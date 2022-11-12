#include "Handle.cuh"

Handle::Handle(const Vertex* vertex, const Vector3f& position) :
    vertex(const_cast<Vertex*>(vertex)),
    position(position) {}

Handle::~Handle() {}

Vertex* Handle::getVertex() const {
    return vertex;
}

Vector3f Handle::getPosition() const {
    return position;
}
