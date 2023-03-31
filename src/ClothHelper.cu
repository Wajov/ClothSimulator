#include "ClothHelper.cuh"

__global__ void initializeHandles(int nHandles, const int* handleIndices, int motionIndex, const Transformation* transformations, Node** nodes, Handle* handles) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nHandles; i += nThreads) {
        Node* node = nodes[handleIndices[i]];
        Handle& handle = handles[i];
        node->preserve = true;
        handle.motionIndex = motionIndex;
        handle.node = node;
        handle.position = motionIndex > -1 ? transformations[motionIndex].inverse().applyToPoint(node->x) : node->x;
    }
}

__global__ void collectHandleIndices(int nHandles, const Handle* handles, int* handleIndices) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nHandles; i += nThreads)
        handleIndices[i] = handles[i].node->index;
}
