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
__global__ void initializeSizing(int nVertices, Vertex** vertices);
__global__ void computeSizingGpu(int nFaces, const Face* const* faces, const Plane* planes, const Remeshing* remeshing);
__global__ void finalizeSizing(int nVertices, Vertex** vertices);
__host__ __device__ bool shouldFlip(const Edge* edge, const Remeshing* remeshing);
__global__ void checkEdgesToFlip(int nEdges, const Edge* const* edges, const Remeshing* remeshing, Edge** edgesToFlip);
__global__ void initializeEdgeNodes(int nEdges, const Edge* const* edges);
__global__ void resetEdgeNodes(int nEdges, const Edge* const* edges);
__global__ void computeEdgeMinIndices(int nEdges, const Edge* const* edges);
__global__ void checkIndependentEdges(int nEdges, const Edge* const* edges, Edge** independentEdges);
__global__ void flipGpu(int nEdges, const Edge* const* edges, const Material* material, Edge** addedEdges, Edge** removedEdges, Face** addedFaces, Face** removedFaces);
__host__ __device__ float edgeMetric(const Vertex* vertex0, const Vertex* vertex1);
__host__ __device__ float edgeMetric(const Edge* edge);
__global__ void checkEdgesToSplit(int nEdges, const Edge* const* edges, Edge** edgesToSplit, float* metrics);
__global__ void splitGpu(int nEdges, const Edge* const* edges, const Material* material, Node** addedNodes, Vertex** addedVertices, Edge** addedEdges, Edge** removedEdges, Face** addedFaces, Face** removedFaces);
__global__ void collectAdjacentEdges(int nEdges, const Edge* const* edges, int* indices, Edge** adjacentEdges);
__global__ void collectAdjacentFaces(int nFaces, const Face* const* faces, int* indices, Face** adjacentFaces);
__global__ void setRange(int n, const int* indices, int* l, int* r);
__device__ bool shouldCollapseGpu(const Edge* edge, int side, const int* edgeBegin, const int* edgeEnd, const Edge* const* adjacentEdges, const int* faceBegin, const int* faceEnd, const Face* const* adjacentFaces, const Remeshing* remeshing);
__global__ void checkEdgesToCollapse(int nEdges, const Edge* const* edges, const int* edgeBegin, const int* edgeEnd, const Edge* const* adjacentEdges, const int* faceBegin, const int* faceEnd, const Face* const* adjacentFaces, const Remeshing* remeshing, Pairei* edgesToCollapse);
__global__ void initializeCollapseNodes(int nEdges, const Pairei* edges, const int* edgeBegin, const int* edgeEnd, const Edge* const* adjacentEdges);
__global__ void resetCollapseNodes(int nEdges, const Pairei* edges, const int* edgeBegin, const int* edgeEnd, const Edge* const* adjacentEdges);
__global__ void computeCollapseMinIndices(int nEdges, const Pairei* edges, const int* edgeBegin, const int* edgeEnd, const Edge* const* adjacentEdges);
__global__ void checkIndependentEdgesToCollapse(int nEdges, const Pairei* edges, const int* edgeBegin, const int* edgeEnd, const Edge* const* adjacentEdges, Pairei* edgesToCollapse);
__global__ void collapseGpu(int nEdges, const Pairei* edges, const Material* material, const int* edgeBegin, const int* edgeEnd, Edge* const* adjacentEdges, const int* faceBegin, const int* faceEnd, Face* const* adjacentFaces, Node** removedNodes, Vertex** removedVertices, Edge** removedEdges, Face** removedFaces);
__global__ void printPlanes(int nPlanes, const Plane* planes);

#endif