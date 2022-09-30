#include "Vertex.hpp"

Vertex::Vertex(int index, const Vector3f& x) :
    index(index),
    x(x),
    n(Vector3f::Zero()),
    u(Vector3f::Zero()),
    v(Vector3f::Zero()),
    m(0.0f) {}

Vertex::~Vertex() {}
