#include "ClothHelper.cuh"

__global__ void initializeHandles(int nHandles, const int* handleIndices, Node** nodes, Handle* handles) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nHandles; i += nThreads) {
        Node* node = nodes[handleIndices[i]];
        Handle& handle = handles[i];
        node->preserve = true;
        handle.node = node;
        handle.position = node->x;
    }
}

__global__ void deleteHandles(int nHandles, const Handle* const* handles) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nHandles; i += nThreads)
        delete handles[i];
}