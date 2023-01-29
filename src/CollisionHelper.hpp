#ifndef COLLISION_HELPER_HPP
#define COLLISION_HELPER_HPP

#include <vector>

#include "MathHelper.cuh"
#include "Vector.cuh"
#include "Face.cuh"
#include "Impact.hpp"

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

static bool checkImpact(ImpactType type, const Vertex* vertex0, const Vertex* vertex1, const Vertex* vertex2, const Vertex* vertex3, Impact& impact) {
    impact.vertices[0] = const_cast<Vertex*>(vertex0);
    impact.vertices[1] = const_cast<Vertex*>(vertex1);
    impact.vertices[2] = const_cast<Vertex*>(vertex2);
    impact.vertices[3] = const_cast<Vertex*>(vertex3);

    Vector3f x0 = vertex0->x0;
    Vector3f v0 = vertex0->x - x0;
    Vector3f x1 = vertex1->x0 - x0;
    Vector3f x2 = vertex2->x0 - x0;
    Vector3f x3 = vertex3->x0 - x0;
    Vector3f v1 = (vertex1->x - vertex1->x0) - v0;
    Vector3f v2 = (vertex2->x - vertex2->x0) - v0;
    Vector3f v3 = (vertex3->x - vertex3->x0) - v0;
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
        Vector3f x0 = vertex0->position(t[i]);
        Vector3f x1 = vertex1->position(t[i]);
        Vector3f x2 = vertex2->position(t[i]);
        Vector3f x3 = vertex3->position(t[i]);

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
    Vertex* vertex0 = face->getVertex(0);
    Vertex* vertex1 = face->getVertex(1);
    Vertex* vertex2 = face->getVertex(2);
    if (vertex == vertex0 || vertex == vertex1 || vertex == vertex2)
        return false;
    if (!vertex->bounds(true).overlap(face->bounds(true), thickness))
        return false;
    
    return checkImpact(VertexFace, vertex, vertex0, vertex1, vertex2, impact);
}

static bool checkEdgeEdgeImpact(const Edge* edge0, const Edge* edge1, float thickness, Impact& impact) {
    Vertex* vertex0 = edge0->getVertex(0);
    Vertex* vertex1 = edge0->getVertex(1);
    Vertex* vertex2 = edge1->getVertex(0);
    Vertex* vertex3 = edge1->getVertex(1);
    if (vertex0 == vertex2 || vertex0 == vertex3 || vertex1 == vertex2 || vertex1 == vertex3)
        return false;
    if (!edge0->bounds(true).overlap(edge1->bounds(true), thickness))
        return false;
    
    return checkImpact(EdgeEdge, vertex0, vertex1, vertex2, vertex3, impact);
}

static void checkImpacts(const Face* face0, const Face* face1, float thickness, std::vector<Impact>& impacts) {
    Impact impact;
    for (int i = 0; i < 3; i++)
        if (checkVertexFaceImpact(face0->getVertex(i), face1, thickness, impact))
            impacts.push_back(impact);
    for (int i = 0; i < 3; i++)
        if (checkVertexFaceImpact(face1->getVertex(i), face0, thickness, impact))
            impacts.push_back(impact);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (checkEdgeEdgeImpact(face0->getEdge(i), face1->getEdge(j), thickness, impact))
                impacts.push_back(impact);
}

#endif