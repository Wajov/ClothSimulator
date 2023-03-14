#include "ObstacleHelper.cuh"

__global__ void setBase(int nNodes, const Node* const* nodes, Vector3f* base) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads)
        base[i] = nodes[i]->x;
}

__global__ void transformGpu(int nNodes, const Vector3f* base, const Transformation transformation, Node** nodes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads) {
        Node* node = nodes[i];
        node->x0 = node->x;
        node->x = transformation.applyToPoint(base[i]);
    }
}
