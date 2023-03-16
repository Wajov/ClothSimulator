#ifndef SEPARATION_HELPER_CUH
#define SEPARATION_HELPER_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "MathHelper.cuh"
#include "Pair.cuh"
#include "Node.cuh"
#include "Face.cuh"
#include "BackupFace.cuh"
#include "Intersection.cuh"

const int MAX_SEPARATION_ITERATION = 100;

__host__ __device__ int majorAxis(const Vector3f& v);
__host__ __device__ bool facePlaneIntersection(const Face* face, const Face* plane, Vector3f& b0, Vector3f& b1);
__host__ __device__ bool checkIntersectionMidpoint(const Face* face0, const Face* face1, Vector3f& b0, Vector3f& b1);
__global__ void checkIntersectionsGpu(int nPairs, const PairFF* pairs, Intersection* intersections);
__global__ void initializeOldPosition(int nIntersections, const Intersection* intersections, Vector3f* x);
__device__ bool containGpu(const Vertex* vertex, int nVertices, const Vertex* const* vertices);
__device__ bool containGpu(const Face* face, int nVertices, const Vertex* const* vertices);
__global__ void collectContainedFaces(int nIntersections, const Intersection* intersections, int nVertices, const Vertex* const* vertices, int* indices, Vector2f* u);
__device__ void oldPositionGpu(const Vector2f& u, const BackupFace& face, Vector3f& x);
__global__ void computeOldPosition(int nIndices, const int* indices, const Vector2f* u, int nFaces, const BackupFace* faces, Vector3f* x);
__host__ __device__ void clearVertexFaceDistance(const Face* face0, const Face* face1, const Vector3f& d, float& maxDist, Vector3f& b0, Vector3f& b1);
__host__ __device__ void clearEdgeEdgeDistance(const Face* face0, const Face* face1, const Vector3f& d, float& maxDist, Vector3f& b0, Vector3f& b1);
__host__ __device__ void farthestPoint(const Face* face0, const Face* face1, const Vector3f& d, Vector3f& b0, Vector3f& b1);
__global__ void computeFarthestPoint(int nIntersections, const Vector3f* x, Intersection* intersections);

#endif