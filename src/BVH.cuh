#ifndef BVH_CUH
#define BVH_CUH

#include <vector>
#include <functional>
#include <unordered_map>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>

#include "CudaHelper.cuh"
#include "BVHHelper.cuh"
#include "BVHNode.cuh"
#include "Node.cuh"
#include "Face.cuh"
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
    void findNearestPoint(const Vector3f& x, NearPoint& point) const;
    void update();
};

#endif