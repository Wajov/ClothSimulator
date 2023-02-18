#ifndef IMPACT_HPP
#define IMPACT_HPP

#include "Vector.cuh"
#include "Node.cuh"

enum ImpactType {
    VertexFace,
    EdgeEdge
};

class Impact {
public:
    Node* nodes[4];
    float t, w[4];
    Vector3f n;
    Impact();
    ~Impact();
    bool operator<(const Impact& impact) const;
};

#endif