#ifndef PHYSICS_HELPER_CUH
#define PHYSICS_HELPER_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "CudaHelper.cuh"
#include "MathHelper.cuh"
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
__global__ void addMass(int nNodes, const Node* const* nodes, Pairii* aIndices, float* aValues);
__global__ void addGravity(int nNodes, const Node* const* nodes, float dt, const Vector3f gravity, int* bIndices, float* bValues);
__global__ void addWindForces(int nFaces, const Face* const* faces, float dt, const Wind* wind, int* bIndices, float* bValues);
__host__ __device__ void stretchingForce(const Face* face, const Material* material, Vector9f& f, Matrix9x9f& J);
__global__ void addStretchingForces(int nFaces, const Face* const* faces, float dt, const Material* material, Pairii* aIndices, float* aValues, int* bIndices, float* bValues);
__host__ __device__ float distance(const Vector3f& x, const Vector3f& a, const Vector3f& b);
__host__ __device__ Vector2f barycentricWeights(const Vector3f& x, const Vector3f& a, const Vector3f& b);
__host__ __device__ void bendingForce(const Edge* edge, const Material* material, Vector12f& f, Matrix12x12f& J);
__global__ void addBendingForces(int nEdges, const Edge* const* edges, float dt, const Material* material, Pairii* aIndices, float* aValues, int* bIndices, float* bValues);
__global__ void addHandleForcesGpu(int nHandles, const Handle* handles, float dt, float stiffness, Pairii* aIndices, float* aValues, int* bIndices, float* bValues);
__host__ __device__ void impulseForce(const Proximity& proximity, float thickness, Vector12f& f, Matrix12x12f& J);
__global__ void splitIndices(int nIndices, const Pairii* indices, int* rowIndices, int* colIndices);
__global__ void setVector(int nIndices, const int* indices, const float* values, float* v);
__global__ void updateNodes(int nNodes, float dt, const float* dv, Node** nodes);

#endif