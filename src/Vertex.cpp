#include "Vertex.hpp"

Vertex::Vertex(int index, const Vector3f& position) :
    index(index),
    position(position),
    normal(Vector3f::Zero()),
    uv(Vector3f::Zero()),
    velocity(Vector3f::Zero()),
    mass(0.0f) {}

Vertex::~Vertex() {}
