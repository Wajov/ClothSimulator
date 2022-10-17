#include "Vertex.hpp"

Vertex::Vertex(int index, const Vector3f& x) :
    index(index),
    x0(x),
    x(x),
    n(Vector3f::Zero()),
    u(Vector3f::Zero()),
    v(Vector3f::Zero()),
    m(0.0f) {}

Vertex::~Vertex() {}

Bounds Vertex::bounds(bool ccd) const {
    Bounds ans;
    ans += x;
    if (ccd)
        ans += x0;
    return ans;
}

Vector3f Vertex::position(float t) const {
    return x0 + t * (x - x0);
}
