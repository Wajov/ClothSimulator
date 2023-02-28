#ifndef OPTIMIZATION_HELPER_CUH
#define OPTIMIZATION_HELPER_CUH

#include <cuda_runtime.h>

#include "MathHelper.cuh"
#include "Vector.cuh"
#include "Node.cuh"
#include "Impact.cuh"

__host__ __device__ float clampViolation(float x, int sign);
__global__ void setDiff(int nNodes, const Node* const* nodes, int* diff);
__global__ void setIndices(int nNodes, const int* nodeIndices, const int* diff, int* indices);
__global__ void initializeGpu(int nNodes, const Node* const* nodes, Vector3f* x);
__global__ void finalizeGpu(int nNodes, const Vector3f* x, Node** nodes);
__global__ void collectCollisionNodes(int nConstraints, const Impact* impacts, int deform, int* indices, Node** nodes);
__global__ void collisionInv(int nNodes, const Node* const* nodes, float obstacleMass, float* inv);
__global__ void collisionObjective(int nNodes, const Node* const* nodes, float obstacleMass, const Vector3f* x, float* objectives);
__global__ void collisionObjectiveGradient(int nNodes, const Node* const* nodes, float invMass, float obstacleMass, const Vector3f* x, Vector3f* gradient);
__global__ void collisionConstraint(int nConstraints, const Impact* impacts, const int* indices, float thickness, const Vector3f* x, float* constraints, int* signs);
__global__ void collectCollisionConstraintGradient(int nConstraints, const Impact* impacts, const float* coefficients, float mu, Vector3f* grad);
__global__ void addConstraintGradient(int n, const int* indices, const Vector3f* grad, Vector3f* gradtient);
__global__ void computeCoefficient(int nConstraints, const float* lambda, float mu, const int* signs, float* c);
__global__ void computeSquare(int nConstraints, float* c);
__global__ void computeNorm2(int nNodes, const Vector3f* x, float* x2);
__global__ void computeXt(int nNodes, const Vector3f* x, const Vector3f* gradient, float s, Vector3f* xt);
__global__ void updateX(int nNodes, const Vector3f* gradient, float s, Vector3f* x);
__global__ void updateMultiplierGpu(int nConstraints, const float* c, const int* signs, float mu, float* lambda);

#endif