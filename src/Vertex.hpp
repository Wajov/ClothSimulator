#ifndef VERTEX_HPP
#define VERTEX_HPP

#include "TypeHelper.hpp"
#include "Bounds.hpp"

class Vertex {
public:
    int index;
    Vector3f x0, x, n, u, v;
    float m;
    Vertex(int index, const Vector3f& x);
    ~Vertex();
    Bounds bounds(bool ccd) const;
    Vector3f position(float t) const;
};

#endif