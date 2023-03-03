#ifndef BVH_NODE_CUH
#define BVH_NODE_CUH

#include <functional>
#include <unordered_map>

#include <cuda_runtime.h>

#include "MathHelper.cuh"
#include "RemeshingHelper.cuh"
#include "Bounds.cuh"
#include "Face.cuh"
#include "NearPoint.cuh"

class BVHNode {
public:
    Face* face;
    Bounds bounds;
    BVHNode* parent, * left, * right;
    int count, maxIndex;
    bool active;
    __host__ __device__ BVHNode();
    __host__ __device__ ~BVHNode();
    void setActiveUp(bool active);
    void setActiveDown(bool active);
    __host__ __device__ bool isLeaf() const;
    void traverse(float thickness, std::function<void(const Face*, const Face*, float)> callback) const;
    void traverse(const BVHNode* node, float thickness, std::function<void(const Face*, const Face*, float)> callback) const;
    void findNearestPoint(const Vector3f& x, NearPoint& point) const;
    void update(bool ccd);
};

#endif