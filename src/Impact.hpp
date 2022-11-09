#ifndef IMPACT_HPP
#define IMPACT_HPP

#include "Vertex.cuh"

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
    bool operator<(const Impact& impact) const;
};

#endif