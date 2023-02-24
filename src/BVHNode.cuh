#ifndef BVH_NODE_CUH
#define BVH_NODE_CUH

// #include <functional>
#include <unordered_map>
#include <cuda_runtime.h>

#include "MathHelper.cuh"
#include "Bounds.cuh"
#include "Face.cuh"
#include "NearPoint.hpp"

class BVHNode {
private:
    float unsignedVertexEdgeDistance(const Vector3f& x, const Vector3f& y0, const Vector3f& y1, Vector3f& n, float& wx, float& wy0, float& wy1) const;
    float unsignedVertexFaceDistance(const Vector3f& x, const Vector3f& y0, const Vector3f& y1, const Vector3f& y2, Vector3f& n, float* w) const;
    void checkNearestPoint(const Vector3f& x, const Face* face, NearPoint& point) const;

public:
    Face* face;
    Bounds bounds;
    BVHNode* parent, * left, * right;
    int count;
    bool active;
    __host__ __device__ BVHNode();
    __host__ __device__ ~BVHNode();
    bool isLeaf() const;
    void setActiveUp(bool active);
    void setActiveDown(bool active);
    void findNearestPoint(const Vector3f& x, NearPoint& point) const;
    void update(bool ccd);
};

#endif