#ifndef SEPARATION_HELPER_CUH
#define SEPARATION_HELPER_CUH

#include <vector>

#include "MathHelper.cuh"
#include "Pair.cuh"
#include "Node.cuh"
#include "Face.cuh"
#include "Intersection.cuh"
#include "Cloth.cuh"

const int MAX_SEPARATION_ITERATION = 100;

__host__ __device__ int majorAxis(const Vector3f& v);
__host__ __device__ bool facePlaneIntersection(const Face* face, const Face* plane, Vector3f& b0, Vector3f& b1);
__host__ __device__ bool checkIntersectionMidpoint(const Face* face0, const Face* face1, Vector3f& b0, Vector3f& b1);
__global__ void checkIntersectionsGpu(int nProximities, const Proximity* proximities, Intersection* intersections);
__host__ __device__ void clearVertexFaceDistance(const Face* face0, const Face* face1, const Vector3f& d, float& maxDist, Vector3f& b0, Vector3f& b1);
__host__ __device__ void clearEdgeEdgeDistance(const Face* face0, const Face* face1, const Vector3f& d, float& maxDist, Vector3f& b0, Vector3f& b1);
__host__ __device__ void farthestPoint(const Face* face0, const Face* face1, const Vector3f& d, Vector3f& b0, Vector3f& b1);

#endif