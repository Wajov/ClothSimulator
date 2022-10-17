#ifndef IMPACT_HPP
#define IMPACT_HPP

#include "Vertex.hpp"

enum ImpactType {
    VertexFace,
    EdgeEdge
};

class Impact {
public:
    Vertex* vertices[4];
    float t, w[4];
    Vector3f n;
    Impact();
    ~Impact();
};

#endif