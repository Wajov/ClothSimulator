#ifndef PHYSICS_HELPER_CUH
#define PHYSICS_HELPER_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "CudaHelper.cuh"
#include "MathHelper.cuh"
#include "MeshHelper.cuh"
#include "Pair.cuh"
#include "Vector.cuh"
#include "Node.cuh"
#include "Vertex.cuh"
#include "Edge.cuh"
#include "Face.cuh"
#include "Wind.cuh"
#include "Handle.cuh"
#include "Material.cuh"
#include "Proximity.cuh"

__host__ __device__ bool inEdge(float w, const Edge* edge0, const Edge* edge1);
__device__ void checkVertexFaceProximityGpu(const Vertex* vertex, const Face* face, PairNi& key0, PairfF& value0, PairFi& key1, PairfN& value1);
__device__ void checkEdgeEdgeProximityGpu(const Edge* edge0, const Edge* edge1, PairEi& key0, PairfE& value0, PairEi& key1, PairfE& value1);
__global__ void checkProximitiesGpu(int nPairs, const PairFF* pairs, PairNi* nodes, PairfF* nodeProximities, PairEi* edges, PairfE* edgeProximities, PairFi* faces, PairfN* faceProximities);
__global__ void setNodeProximities(int nProximities, const PairNi* nodes, const PairfF* nodeProximities, float thickness, float stiffness, float clothFriction, float obstacleFriction, Proximity* proximities);
__global__ void setEdgeProximities(int nProximities, const PairEi* edges, const PairfE* edgeProximities, float thickness, float stiffness, float clothFriction, float obstacleFriction, Proximity* proximities);
__global__ void setFaceProximities(int nProximities, const PairFi* faces, const PairfN* faceProximities, float thickness, float stiffness, float clothFriction, float obstacleFriction, Proximity* proximities);
__global__ void addMass(int nNodes, const Node* const* nodes, Pairii* aIndices, float* aValues);
__global__ void addGravity(int nNodes, const Node* const* nodes, float dt, const Vector3f gravity, int* bIndices, float* bValues);
__global__ void addWindForces(int nFaces, const Face* const* faces, float dt, const Wind* wind, int* bIndices, float* bValues);
__device__ void addMatrixAndVectorGpu(const Matrix9x9f& B, const Vector9f& b, const Vector3i& indices, Pairii* aIndices, float* aValues, int* bIndices, float* bValues);
__device__ void addMatrixAndVectorGpu(const Matrix12x12f& B, const Vector12f& b, const Vector4i& indices, Pairii* aIndices, float* aValues, int* bIndices, float* bValues);
__host__ __device__ void stretchingForce(const Face* face, const Material* material, Vector9f& f, Matrix9x9f& J);
__global__ void addStretchingForces(int nFaces, const Face* const* faces, float dt, const Material* material, Pairii* aIndices, float* aValues, int* bIndices, float* bValues);
__host__ __device__ float distance(const Vector3f& x, const Vector3f& a, const Vector3f& b);
__host__ __device__ Vector2f barycentricWeights(const Vector3f& x, const Vector3f& a, const Vector3f& b);
__host__ __device__ void bendingForce(const Edge* edge, const Material* material, Vector12f& f, Matrix12x12f& J);
__global__ void addBendingForces(int nEdges, const Edge* const* edges, float dt, const Material* material, Pairii* aIndices, float* aValues, int* bIndices, float* bValues);
__global__ void addHandleForcesGpu(int nHandles, const Handle* handles, float dt, float stiffness, Pairii* aIndices, float* aValues, int* bIndices, float* bValues);
__host__ __device__ void impulseForce(const Proximity& proximity, float d, float thickness, Vector12f& f, Matrix12x12f& J);
__host__ __device__ void frictionForce(const Proximity& proximity, float d, float thickness, float dt, Vector12f& f, Matrix12x12f& J);
__global__ void addProximityForcesGpu(int nProximities, const Proximity* proximities, float dt, float thickness, int nNodes, const Node* const* nodes, Pairii* aIndices, float* aValues, int* bIndices, float* bValues);
__global__ void splitIndices(int nIndices, const Pairii* indices, int* rowIndices, int* colIndices);
__global__ void setVector(int nIndices, const int* indices, const float* values, float* v);
__global__ void updateNodes(int nNodes, float dt, const float* dv, Node** nodes);

#endif