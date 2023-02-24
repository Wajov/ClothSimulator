#ifndef COLLISION_HELPER_CUH
#define COLLISION_HELPER_CUH

#include <vector>

#include "MathHelper.cuh"
#include "Node.cuh"
#include "Vector.cuh"
#include "Face.cuh"
#include "Impact.cuh"

const int MAX_COLLISION_ITERATION = 100;

static float signedVertexFaceDistance(const Vector3f& x, const Vector3f& y0, const Vector3f& y1, const Vector3f& y2, Vector3f& n, float* w) {
    n = (y1 - y0).normalized().cross((y2 - y0).normalized());
    if (n.norm2() < 1e-6f)
        return INFINITY;
    n.normalize();
    float h = (x - y0).dot(n);
    float b0 = mixed(y1 - x, y2 - x, n);
    float b1 = mixed(y2 - x, y0 - x, n);
    float b2 = mixed(y0 - x, y1 - x, n);
    w[0] = 1.0f;
    w[1] = -b0 / (b0 + b1 + b2);
    w[2] = -b1 / (b0 + b1 + b2);
    w[3] = -b2 / (b0 + b1 + b2);
    return h;
}

static float signedEdgeEdgeDistance(const Vector3f& x0, const Vector3f& x1, const Vector3f& y0, const Vector3f& y1, Vector3f& n, float* w) {
    n = (x1 - x0).normalized().cross((y1 - y0).normalized());
    if (n.norm2() < 1e-8f) {
        Vector3f e0 = (x1 - x0).normalized(), e1 = (y1 - y0).normalized();
        float p0min = x0.dot(e0), p0max = x1.dot(e0), p1min = y0.dot(e0), p1max = y1.dot(e0);
        if (p1max < p1min)
            mySwap(p1max, p1min);
        
        float a = max(p0min, p1min), b = min(p0max, p1max), c = 0.5f * (a + b);
        if (a > b)
            return INFINITY;
        
        Vector3f d = y0 - x0 - (y0-x0).dot(e0) * e0;
        n = (-d).normalized();
        w[1] = (c - x0.dot(e0)) / (x1 - x0).norm();
        w[0] = 1.0f - w[1];
        w[3] = -(e0.dot(e1) * c - y0.dot(e1)) / (y1-y0).norm();
        w[2] = -1.0f - w[3];
        return d.norm();
    }
    n = n.normalized();
    float h = (x0 - y0).dot(n);
    float a0 = mixed(y1 - x1, y0 - x1, n);
    float a1 = mixed(y0 - x0, y1 - x0, n);
    float b0 = mixed(x0 - y1, x1 - y1, n);
    float b1 = mixed(x1 - y0, x0 - y0, n);
    w[0] = a0 / (a0 + a1);
    w[1] = a1 / (a0 + a1);
    w[2] = -b0 / (b0 + b1);
    w[3] = -b1 / (b0 + b1);
    return h;
}

static bool checkImpact(ImpactType type, const Node* node0, const Node* node1, const Node* node2, const Node* node3, Impact& impact) {
    impact.nodes[0] = const_cast<Node*>(node0);
    impact.nodes[1] = const_cast<Node*>(node1);
    impact.nodes[2] = const_cast<Node*>(node2);
    impact.nodes[3] = const_cast<Node*>(node3);

    Vector3f x0 = node0->x0;
    Vector3f v0 = node0->x - x0;
    Vector3f x1 = node1->x0 - x0;
    Vector3f x2 = node2->x0 - x0;
    Vector3f x3 = node3->x0 - x0;
    Vector3f v1 = (node1->x - node1->x0) - v0;
    Vector3f v2 = (node2->x - node2->x0) - v0;
    Vector3f v3 = (node3->x - node3->x0) - v0;
    float a0 = mixed(x1, x2, x3);
    float a1 = mixed(v1, x2, x3) + mixed(x1, v2, x3) + mixed(x1, x2, v3);
    float a2 = mixed(x1, v2, v3) + mixed(v1, x2, v3) + mixed(v1, v2, x3);
    float a3 = mixed(v1, v2, v3);

    if (abs(a0) < 1e-6f * x1.norm() * x2.norm() * x3.norm())
        return false;

    float t[3];
    int nSolution = solveCubic(a3, a2, a1, a0, t);
    for (int i = 0; i < nSolution; i++) {
        if (t[i] < 0.0f || t[i] > 1.0f)
            continue;
        impact.t = t[i];
        Vector3f x0 = node0->position(t[i]);
        Vector3f x1 = node1->position(t[i]);
        Vector3f x2 = node2->position(t[i]);
        Vector3f x3 = node3->position(t[i]);

        Vector3f& n = impact.n;
        float* w = impact.w;
        float d;
        bool inside;
        if (type == VertexFace) {
            d = signedVertexFaceDistance(x0, x1, x2, x3, n, w);
            inside = (min(-w[1], -w[2], -w[3]) >= -1e-6f);
        } else {
            d = signedEdgeEdgeDistance(x0, x1, x2, x3, n, w);
            inside = (min(w[0], w[1], -w[2], -w[3]) >= -1e-6f);
        }
        if (n.dot(w[1] * v1 + w[2] * v2 + w[3] * v3) > 0.0f)
            n = -n;
        if (abs(d) < 1e-6f && inside)
            return true;
    }
    return false;
}

static bool checkVertexFaceImpact(const Vertex* vertex, const Face* face, float thickness, Impact& impact) {
    Node* node = vertex->node;
    Node* node0 = face->vertices[0]->node;
    Node* node1 = face->vertices[1]->node;
    Node* node2 = face->vertices[2]->node;
    if (node == node0 || node == node1 || node == node2)
        return false;
    if (!node->bounds(true).overlap(face->bounds(true), thickness))
        return false;
    
    return checkImpact(VertexFace, node, node0, node1, node2, impact);
}

static bool checkEdgeEdgeImpact(const Edge* edge0, const Edge* edge1, float thickness, Impact& impact) {
    Node* node0 = edge0->nodes[0];
    Node* node1 = edge0->nodes[1];
    Node* node2 = edge1->nodes[0];
    Node* node3 = edge1->nodes[1];
    if (node0 == node2 || node0 == node3 || node1 == node2 || node1 == node3)
        return false;
    if (!edge0->bounds(true).overlap(edge1->bounds(true), thickness))
        return false;
    
    return checkImpact(EdgeEdge, node0, node1, node2, node3, impact);
}

static void checkImpacts(const Face* face0, const Face* face1, float thickness, std::vector<Impact>& impacts) {
    Impact impact;
    for (int i = 0; i < 3; i++)
        if (checkVertexFaceImpact(face0->vertices[i], face1, thickness, impact))
            impacts.push_back(impact);
    for (int i = 0; i < 3; i++)
        if (checkVertexFaceImpact(face1->vertices[i], face0, thickness, impact))
            impacts.push_back(impact);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (checkEdgeEdgeImpact(face0->edges[i], face1->edges[j], thickness, impact))
                impacts.push_back(impact);
}

#endif