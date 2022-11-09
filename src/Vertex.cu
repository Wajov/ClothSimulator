#include "Vertex.cuh"

Vertex::Vertex(const Vector3f& x, bool isFree) :
    index(0),
    x0(x),
    x1(x),
    x(x),
    n(0.0f, 0.0f, 0.0f),
    v(0.0f, 0.0f, 0.0f),
    u(0.0f, 0.0f),
    sizing(),
    m(0.0f),
    a(0.0f),
    isFree(isFree),
    preserve(false) {}

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
