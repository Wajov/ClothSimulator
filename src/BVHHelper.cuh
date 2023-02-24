#ifndef BVH_HELPER_CUH
#define BVH_HELPER_CUH


#include <device_launch_parameters.h>

#include "Face.cuh"
#include "Bounds.cuh"
#include "BVHNode.cuh"

__global__ static void computeLeafBounds(int nFaces, const Face* const* faces, bool ccd, Bounds* bounds) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads)
        bounds[i] = faces[i]->bounds(ccd);
}

__device__ static unsigned int expandBits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ static unsigned int mortonCode(const Vector3f& v, const Vector3f& p, const Vector3f& d) {
    float x = d(0) == 0.0f ? 0.0f : (v(0) - p(0)) / d(0);
    float y = d(1) == 0.0f ? 0.0f : (v(1) - p(1)) / d(1);
    float z = d(2) == 0.0f ? 0.0f : (v(2) - p(2)) / d(2);
    x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return (xx << 2) + (yy << 1) + zz;
}

__global__ static void initializeLeafNodes(int nNodes, const Face* const* faces, const Bounds* bounds, const Vector3f p, const Vector3f d, BVHNode* nodes, unsigned long long* mortonCodes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads) {
        const Bounds& b = bounds[i];
        BVHNode& node = nodes[i];
        node.face = const_cast<Face*>(faces[i]);
        node.bounds = b;
        unsigned long long code = mortonCode(b.center(), p, d);
        mortonCodes[i] = (code << 32) | static_cast<unsigned long long>(i);
    }
}

__device__ static int commonUpperBits(unsigned long long a, unsigned long long b) {
    return __clzll(a ^ b);
}

__device__ static void findRange(int nNodes, const unsigned long long* mortonCodes, int i, int& left, int& right) {
    if (i == 0) {
        left = 0;
        right = nNodes;
        return;
    }

    unsigned long long code = mortonCodes[i];
    int dl = commonUpperBits(code, mortonCodes[i - 1]);
    int dr = commonUpperBits(code, mortonCodes[i + 1]);
    int l, r, mid;
    if (dl < dr) {
        l = -1;
        r = i;
        while (r - l > 1) {
            mid = (l + r) >> 1;
            commonUpperBits(code, mortonCodes[mid]) < dr ? r = mid : l = mid;
        }
        left = r;
        right = i;
    } else {
        l = i;
        r = nNodes + 1;
        while (r - l > 1) {
            commonUpperBits(code, mortonCodes[mid] < dl ? l = mid : r = mid);
        }
        left = i;
        right = l;
    }
}

__device__ static int findSplit(const unsigned long long* mortonCodes, int left, int right) {
    unsigned long long code = mortonCodes[left];
    int d = commonUpperBits(code, mortonCodes[right]);
    int l = left, r = right, mid;
    while (r - l > 1) {
        mid = (l + r) >> 1;
        commonUpperBits(code, mortonCodes[mid]) > d ? l = mid : r = mid;
    }
    return l;
}

__global__ static void initializeInternalNodes(int nNodes, const unsigned long long* mortonCodes, const BVHNode* leaves, BVHNode* internals) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads) {
        BVHNode& node = internals[i];
        int left, right;
        findRange(nNodes, mortonCodes, i, left, right);
        int middle = findSplit(mortonCodes, left, right);
        node.left = left == middle ? const_cast<BVHNode*>(&leaves[middle]) : &internals[middle];
        node.right = middle + 1 == right ? const_cast<BVHNode*>(&leaves[middle + 1]) : &internals[middle + 1];
        node.left->parent = node.right->parent = &node;
    }
}

__global__ static void computeInternalBounds(int nNodes, BVHNode* nodes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads) {
        BVHNode* node = nodes[i].parent;
        while (node != nullptr) {
            if (atomicAdd(&node->count, 1) == 1) {
                node->bounds = node->left->bounds + node->right->bounds;
                break;
            }
            node = node->parent;
        }
    }
}

#endif