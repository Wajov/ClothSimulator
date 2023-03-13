#include "Node.cuh"

Node::Node(const Vector3f& x, bool isFree) :
    index(0),
    x0(x),
    x1(x),
    x(x),
    isFree(isFree),
    preserve(false) {}

Node::~Node() {}

Bounds Node::bounds(bool ccd) const {
    Bounds ans;
    ans += x;
    if (ccd)
        ans += x0;
    return ans;
}

Vector3f Node::position(float t) const {
    return x0 + t * (x - x0);
}