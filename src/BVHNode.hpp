#ifndef BVH_NODE_HPP
#define BVH_NODE_HPP

#include <cmath>
#include <functional>
#include <unordered_map>

#include "MathHelper.cuh"
#include "CollisionHelper.hpp"
#include "Bounds.hpp"
#include "Face.cuh"
#include "NearPoint.hpp"

class BVHNode {
private:
    Face* face;
    Bounds bounds;
    BVHNode* parent, * left, * right;
    bool active;
    float unsignedVertexEdgeDistance(const Vector3f& x, const Vector3f& y0, const Vector3f& y1, Vector3f& n, float& wx, float& wy0, float& wy1) const;
    float unsignedVertexFaceDistance(const Vector3f& x, const Vector3f& y0, const Vector3f& y1, const Vector3f& y2, Vector3f& n, float* w) const;
    void checkNearestPoint(const Vector3f& x, const Face* face, NearPoint& point) const;

public:
    BVHNode(BVHNode* parent, int l, int r, std::vector<Face*>& faces, std::vector<Bounds>& bounds, std::vector<Vector3f>& centers, std::unordered_map<Face*, BVHNode*>& leaves);
    ~BVHNode();
    inline bool isLeaf() const;
    void setActiveUp(bool active);
    void setActiveDown(bool active);
    void traverse(float thickness, std::function<void(const Face*, const Face*, float)> callback);
    void traverse(const BVHNode* bvhNode, float thickness, std::function<void(const Face*, const Face*, float)> callback);
    void findNearestPoint(const Vector3f& x, NearPoint& point) const;
    void update(bool ccd);
};

#endif