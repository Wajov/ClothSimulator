#ifndef SEPARATION_HELPER_CUH
#define SEPARATION_HELPER_CUH

#include <vector>

#include "MathHelper.cuh"
#include "Node.cuh"
#include "Face.cuh"
#include "Intersection.cuh"
#include "Cloth.cuh"

const int MAX_SEPARATION_ITERATION = 100;

static int majorAxis(const Vector3f& v) {
    return abs(v(0)) > abs(v(1)) && abs(v(0)) > abs(v(2)) ? 0 : (abs(v(1)) > abs(v(2)) ? 1 : 2);
}

static bool facePlaneIntersection(const Face* face, const Face* plane, Vector3f& b0, Vector3f& b1) {
    Vector3f x0 = plane->vertices[0]->node->x;
    Vector3f n = plane->n;
    float h[3];
    int signSum = 0;
    for (int i = 0; i < 3; i++) {
        h[i] = (face->vertices[i]->node->x - x0).dot(n);
        signSum += sign(h[i]);
    }
    if (signSum == -3 || signSum == 3)
        return false;
    
    int v0 = -1;
    for (int i = 0; i < 3; i++)
        if (sign(h[i]) == -signSum)
            v0 = i;
    int v1 = (v0 + 1) % 3;
    int v2 = (v0 + 2) % 3;
    float t0 = h[v0] / (h[v0] - h[v1]);
    float t1 = h[v0] / (h[v0] - h[v2]);
    b0(v0) = 1.0f - t0;
    b0(v1) = t0;
    b0(v2) = 0.0f;
    b1(v0) = 1.0f - t1;
    b1(v2) = t1;
    b1(v1) = 0.0f;
    return true;
}

static bool intersectionMidpoint(const Face* face0, const Face* face1, Vector3f& b0, Vector3f& b1) {
    if (face0->adjacent(face1))
        return false;
    
    Vector3f c = face0->n.cross(face1->n);
    if (c.norm2() < 1e-10f)
        return false;
    
    Vector3f b00, b01, b10, b11;
    bool flag0 = facePlaneIntersection(face0, face1, b00, b01);
    bool flag1 = facePlaneIntersection(face1, face0, b10, b11);
    if (!flag0 || !flag1)
        return false;
    
    int axis = majorAxis(c);
    float a00 = face0->position(b00)(axis);
    float a01 = face0->position(b01)(axis);
    float a10 = face1->position(b10)(axis);
    float a11 = face1->position(b11)(axis);
    float aMin = max(min(a00, a01), min(a10, a11));
    float aMax = min(max(a00, a01), max(a10, a11));
    if (aMin > aMax)
        return false;
    
    float aMid = 0.5f * (aMin + aMax);
    b0 = (a00 == a01) ? b00 : b00 + (aMid - a00) / (a01 - a00) * (b01 - b00);
    b1 = (a10 == a11) ? b10 : b10 + (aMid - a10) / (a11 - a10) * (b11 - b10);
    return true;
}

static Vector3f oldPosition(const Face* face, const Vector3f& b, const std::vector<Cloth*>& cloths, const std::vector<Mesh*>& oldMeshes) {
    if (!face->isFree())
        return face->position(b);
    
    Vector2f u = b(0) * face->vertices[0]->u + b(1) * face->vertices[1]->u + b(2) * face->vertices[2]->u;
    for (int i = 0; i < cloths.size(); i++)
        if (cloths[i]->getMesh()->contain(face))
            return oldMeshes[i]->oldPosition(u);
}

static void clearVertexFaceDistance(const Face* face0, const Face* face1, const Vector3f& d, float& maxDist, Vector3f& b0, Vector3f& b1) {
    Vector3f x0 = face1->vertices[0]->node->x;
    Vector3f x1 = face1->vertices[1]->node->x;
    Vector3f x2 = face1->vertices[2]->node->x;
    Vector3f n = face1->n;
    for (int i = 0; i < 3; i++) {
        Vector3f x = face0->vertices[i]->node->x;
        
        float h = (x - x0).dot(n);
        float dh = d.dot(n);
        if (h * dh >= 0.0f)
            continue;
        
        float a0 = mixed(x2 - x1, x - x1, d);
        float a1 = mixed(x0 - x2, x - x2, d);
        float a2 = mixed(x1 - x0, x - x0, d);
        if (a0 <= 0.0f || a1 <= 0.0f || a2 <= 0.0f)
            continue;
        
        float dist = -h / dh;
        if (dist > maxDist) {
            maxDist = dist;
            b0 = Vector3f();
            b0(i) = 1.0f;
            b1(0) = a0 / (a0 + a1 + a2);
            b1(1) = a1 / (a0 + a1 + a2);
            b1(2) = a2 / (a0 + a1 + a2);
        }
    }
}

static void clearEdgeEdgeDistance(const Face* face0, const Face* face1, const Vector3f& d, float& maxDist, Vector3f& b0, Vector3f& b1) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            Vector3f x00 = face0->vertices[i]->node->x;
            Vector3f x01 = face0->vertices[(i + 1) % 3]->node->x;
            Vector3f x10 = face1->vertices[j]->node->x;
            Vector3f x11 = face1->vertices[(j + 1) % 3]->node->x;
            Vector3f n = (x01 - x00).cross(x11 - x10).normalized();
            
            float h = (x00 - x10).dot(n);
            float dh = d.dot(n);
            if (h * dh >= 0.0f)
                continue;
            
            float a00 = mixed(x01 - x10, x11 - x10, d);
            float a01 = mixed(x11 - x10, x00 - x10, d);
            float a10 = mixed(x01 - x00, x11 - x00, d);
            float a11 = mixed(x10 - x00, x01 - x00, d);
            if (a00 * a01 <= 0.0f || a10 * a11 <= 0.0f)
                continue;
            
            float dist = -h / dh;
            if (dist > maxDist) {
                maxDist = dist;
                b0 = Vector3f();
                b0(i) = a00 / (a00 + a01);
                b0((i + 1) % 3) = a01 / (a00 + a01);
                b1 = Vector3f();
                b1(j) = a10 / (a10 + a11);
                b1((j + 1) % 3) = a11 / (a10 + a11);
            }
        }
}

static void farthestPoint(const Face* face0, const Face* face1, const Vector3f& d, Vector3f& b0, Vector3f& b1) {
    float maxDist = 0.0f;
    clearVertexFaceDistance(face0, face1, d, maxDist, b0, b1);
    clearVertexFaceDistance(face1, face0, -d, maxDist, b1, b0);
    clearEdgeEdgeDistance(face0, face1, d, maxDist, b0, b1);
}

static void checkIntersection(const Face* face0, const Face* face1, std::vector<Intersection>& intersections, const std::vector<Cloth*>& cloths, const std::vector<Mesh*>& oldMeshes) {
    Vector3f b0, b1;
    if (intersectionMidpoint(face0, face1, b0, b1)) {
        Vector3f x0 = oldPosition(face0, b0, cloths, oldMeshes);
        Vector3f x1 = oldPosition(face1, b1, cloths, oldMeshes);
        Vector3f d = (x0 - x1).normalized();
        farthestPoint(face0, face1, d, b0, b1);
        intersections.emplace_back(face0, face1, b0, b1, d);
    }
}

#endif