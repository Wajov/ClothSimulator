#ifndef BVH_CUH
#define BVH_CUH

#include <vector>
#include <functional>
#include <unordered_map>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>

#include "CudaHelper.cuh"
#include "BVHHelper.cuh"
#include "BVHNode.cuh"
#include "Node.cuh"
#include "Face.cuh"
#include "Bounds.cuh"
#include "Mesh.cuh"
#include "Impact.cuh"
#include "NearPoint.cuh"

extern bool gpu;

class BVH {
private:
    int index;
    bool ccd;
    BVHNode* root;
    std::vector<BVHNode> nodes;
    std::unordered_map<Node*, std::vector<BVHNode*>> adjacents;
    thrust::device_vector<BVHNode> leaves, internals;

public:
    BVH(const Mesh* mesh, bool ccd);
    ~BVH();
    void initialize(const BVHNode* parent, int l, int r, std::vector<Face*>& faces, std::vector<Bounds>& bounds, std::vector<Vector3f>& centers);
    bool contain(const Node* node) const;
    void setAllActive(bool active);
    void setActive(const Node* node, bool active);
    void traverse(float thickness, std::function<void(const Face*, const Face*, float)> callback) const;
    void traverse(const BVH* bvh, float thickness, std::function<void(const Face*, const Face*, float)> callback) const;
    thrust::device_vector<Proximity> traverse(float thickness) const;
    thrust::device_vector<Proximity> traverse(const BVH* bvh, float thickness) const;
    void findNearestPoint(const Vector3f& x, NearPoint& point) const;
    void findNearestPoint(const thrust::device_vector<Vector3f>& x, thrust::device_vector<NearPoint>& points) const;
    void update();
};

#endif