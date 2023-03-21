#ifndef BVH_HELPER_CUH
#define BVH_HELPER_CUH

#include <device_launch_parameters.h>

#include "RemeshingHelper.cuh"
#include "Pair.cuh"
#include "Face.cuh"
#include "Bounds.cuh"
#include "BVHNode.cuh"

__global__ void computeLeafBounds(int nFaces, const Face* const* faces, bool ccd, Bounds* bounds);
__device__ unsigned int expandBits(unsigned int v);
__device__ unsigned int mortonCode(const Vector3f& v, const Vector3f& p, const Vector3f& d);
__global__ void initializeLeafNodes(int nNodes, const Face* const* faces, const Bounds* bounds, const Vector3f p, const Vector3f d, BVHNode* nodes, unsigned long long* mortonCodes);
__device__ int commonUpperBits(unsigned long long a, unsigned long long b);
__device__ void findRange(int nNodes, const unsigned long long* mortonCodes, int i, int& left, int& right);
__device__ int findSplit(const unsigned long long* mortonCodes, int left, int right);
__global__ void initializeInternalNodes(int nNodes, const unsigned long long* mortonCodes, BVHNode* leaves, BVHNode* internals);
__device__ float atomicMin(float* address, float val);
__device__ float atomicMax(float* address, float val);
__global__ void computeInternalBounds(int nNodes, BVHNode* nodes);
__global__ void countPairsSelf(int nLeaves, const BVHNode* leaves, const BVHNode* root, float thickness, int* num);
__global__ void findPairsSelf(int nLeaves, const BVHNode* leaves, const BVHNode* root, float thickness, const int* num, PairFF* pairs);
__global__ void countPairs(int nLeaves, const BVHNode* leaves, const BVHNode* root, float thickness, int* num);
__global__ void findPairs(int nLeaves, const BVHNode* leaves, const BVHNode* root, float thickness, const int* num, PairFF* pairs);
__global__ void findNearestPointGpu(int nNodes, const Vector3f* x, const BVHNode* root, NearPoint* points);
__global__ void resetCount(int nNodes, BVHNode* nodes);
__global__ void updateGpu(int nNodes, BVHNode* nodes, bool ccd);

#endif