#ifndef VERTEX_HPP
#define VERTEX_HPP

#include "TypeHelper.hpp"

class Vertex {
public:
    int index;
    Vector3f position, normal, uv, velocity;
    float mass;
    Vertex(int index, const Vector3f& position);
    ~Vertex();
};

#endif