#ifndef COLLISION_HELPER_CUH
#define COLLISION_HELPER_CUH

#include <vector>

#include <device_launch_parameters.h>

#include "MathHelper.cuh"
#include "BVHHelper.cuh"
#include "Pair.cuh"
#include "Node.cuh"
#include "Vector.cuh"
#include "Face.cuh"
#include "Impact.cuh"

enum ImpactType {
    VertexFace,
    EdgeEdge
};

const int MAX_COLLISION_ITERATION = 100;

__host__ __device__ float newtonsMethod(float a, float b, float c, float d, float x0, int dir);
__host__ __device__ int solveQuadratic(float a, float b, float c, float x[2]);
__host__ __device__ int solveCubic(float a, float b, float c, float d, float x[]);
__host__ __device__ bool checkImpact(ImpactType type, const Node* node0, const Node* node1, const Node* node2, const Node* node3, Impact& impact);
__host__ __device__ bool checkVertexFaceImpact(const Vertex* vertex, const Face* face, float thickness, Impact& impact);
__host__ __device__ bool checkEdgeEdgeImpact(const Edge* edge0, const Edge* edge1, float thickness, Impact& impact);
__global__ void checkImpactsGpu(int nPairs, const PairFF* pairs, float thickness, Impact* impacts);
__global__ void initializeImpactNodes(int nImpacts, const Impact* impacts, int deform);
__global__ void collectRelativeImpacts(int nImpacts, const Impact* impacts, int deform, Node** nodes, int* relativeImpacts);
__global__ void setImpactMinIndices(int nNodes, const int* relativeImpacts, Node** nodes);
__global__ void checkIndependentImpacts(int nImpacts, const Impact* impacts, int deform, Impact* independentImpacts);

#endif