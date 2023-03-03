#ifndef REMESHING_HELPER_CUH
#define REMESHING_HELPER_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "MathHelper.cuh"
#include "CudaHelper.cuh"
#include "NearPoint.cuh"
#include "Pair.cuh"
#include "Vector.cuh"
#include "Matrix.cuh"
#include "Vertex.cuh"
#include "Face.cuh"
#include "Plane.cuh"
#include "Disk.cuh"
#include "Remeshing.cuh"

__global__ void setX(int nNodes, const Node* const* nodes, Vector3f* x);
__global__ void initializeNearPoints(int nNodes, const Vector3f* x, NearPoint* points);
__host__ __device__ float unsignedVertexEdgeDistance(const Vector3f& x, const Vector3f& y0, const Vector3f& y1, Vector3f& n, float& wx, float& wy0, float& wy1);
__host__ __device__ float unsignedVertexFaceDistance(const Vector3f& x, const Vector3f& y0, const Vector3f& y1, const Vector3f& y2, Vector3f& n, float* w);
__host__ __device__ void checkNearestPoint(const Vector3f& x, const Face* face, NearPoint& point);
__global__ void setNearestPlane(int nNodes, const Vector3f* x, const NearPoint* points, Plane* planes);
__host__ __device__ Matrix2x2f diagonal(const Vector2f& v);
__host__ __device__ Matrix2x2f sqrt(const Matrix2x2f& A);
__host__ __device__ Matrix2x2f max(const Matrix2x2f& A, float v);
__host__ __device__ Matrix2x2f compressionMetric(const Matrix2x2f& G, const Matrix2x2f& S2, const Remeshing* remeshing);
__host__ __device__ Matrix2x2f obstacleMetric(const Face* face, const Plane* planes);
__host__ __device__ Disk circumscribedDisk(const Disk& d0, const Disk& d1);
__host__ __device__ Disk circumscribedDisk(const Disk& d0, const Disk& d1, const Disk& d2);
__host__ __device__ Matrix2x2f maxTensor(const Matrix2x2f* M);
__host__ __device__ Matrix2x2f faceSizing(const Face* face, const Plane* planes, const Remeshing* remeshing);
__global__ void collectVertexSizing(int nFaces, const Face* const* faces, const Plane* planes, int* indices, Pairfm* sizing, const Remeshing* remeshing);
__global__ void setVertexSizing(int nIndices, const int* indices, const Pairfm* sizing, Vertex** vertices);

#endif