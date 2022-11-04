#ifndef VERTEX_HPP
#define VERTEX_HPP

#include "TypeHelper.hpp"
#include "Bounds.hpp"

class Vertex {
public:
    int index;
    Vector3f x0, x1, x, n, v;
    Vector2f u;
    Matrix2x2f sizing;
    float m, a;
    bool isFree, preserve;
    Vertex(int index, const Vector3f& x, bool isFree);
    ~Vertex();
    Bounds bounds(bool ccd) const;
    Vector3f position(float t) const;
};

#endif