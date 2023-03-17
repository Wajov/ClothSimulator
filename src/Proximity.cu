#include "Proximity.cuh"

Proximity::Proximity() {}

Proximity::Proximity(const Node* node, const Face* face, float stiffness, float clothFriction, float obstacleFriction) :
    nodes{const_cast<Node*>(node), face->vertices[0]->node, face->vertices[1]->node, face->vertices[2]->node},
    stiffness(stiffness * min(node->area, face->area)) {
    float d = signedVertexFaceDistance(nodes[0]->x, nodes[1]->x, nodes[2]->x, nodes[3]->x, n, w);
    if (d < 0.0f)
        n = -n;
    mu = node->isFree && face->isFree() ? clothFriction : obstacleFriction;
}

Proximity::Proximity(const Edge* edge0, const Edge* edge1, float stiffness, float clothFriction, float obstacleFriction) :
    nodes{edge0->nodes[0], edge0->nodes[1], edge1->nodes[0], edge1->nodes[1]},
    stiffness(stiffness * min(edge0->area(), edge1->area())) {
    float d = signedEdgeEdgeDistance(nodes[0]->x, nodes[1]->x, nodes[2]->x, nodes[3]->x, n, w);
    if (d < 0.0f)
        n = -n;
    mu = edge0->isFree() && edge1->isFree() ? clothFriction : obstacleFriction;
}

Proximity::~Proximity() {}