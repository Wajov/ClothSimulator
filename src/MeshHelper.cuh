#ifndef MESH_HELPER_CUH
#define MESH_HELPER_CUH

#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>

#include "CudaHelper.cuh"
#include "MathHelper.cuh"
#include "Pair.cuh"
#include "Node.cuh"
#include "Vertex.cuh"
#include "Edge.cuh"
#include "Face.cuh"
#include "Renderable.cuh"
#include "Material.cuh"
#include "BackupFace.cuh"

struct EdgeData {
    Vertex* opposite;
    Face* adjacent;
};

struct NodeData {
    float mass, area;

    __device__ NodeData operator+(const NodeData& d) const {
        NodeData ans;
        ans.mass = mass + d.mass;
        ans.area = area + d.area;
        return ans;
    };
};

__global__ void initializeNodes(int nNodes, const Vector3f* x, bool isFree, int nVelocities, const Vector3f* v, Node** nodes);
__global__ void initializeVertices(int nVertices, const Vector2f* u, Vertex** vertices);
__device__ void setEdgeData(int index0, int index1, const Vertex* vertex, const Face* face, Pairii& index, EdgeData& edgeData);
__global__ void initializeFaces(int nFaces, const int* xIndices, const int* uIndices, const Node* const* nodes, const Material* material, Vertex** vertices, Face** faces, Pairii* edgeIndices, EdgeData* edgeData);
__global__ void initializeEdges(int nEdges, const Pairii* indices, const EdgeData* edgeData, const Node* const* nodes, Edge** edges);
__global__ void setEdges(int nEdges, const Pairii* indices, const EdgeData* edgeData, Edge** edges);
__global__ void setPreserve(int nEdges, const Edge* const* edges);
__device__ bool containGpu(const Node* node, int nNodes, const Node* const* nodes);
__device__ bool containGpu(const Vertex* vertex, int nVertices, const Vertex* const* vertices);
__device__ bool containGpu(const Face* face, int nVertices, const Vertex* const* vertices);
__global__ void setBackupFaces(int nFaces, const Face* const* faces, BackupFace* backupFaces);
__global__ void initializeIndices(int n, int* indices);
__global__ void updateNodeIndices(int nNodes, Node** nodes);
__global__ void updateVertexIndices(int nVertices, Vertex** vertices);
__global__ void initializeNodeGeometries(int nNodes, Node** nodes);
__global__ void updateNodeGeometriesGpu(int nFaces, const Face* const* faces);
__global__ void finalizeNodeGeometries(int nNodes, Node** nodes);
__global__ void updateFaceGeometriesGpu(int nFaces, Face** faces);
__global__ void updatePositionsGpu(int nNodes, float dt, Node** nodes);
__global__ void updateVelocitiesGpu(int nNodes, float invDt, Node** nodes);
__global__ void updateRenderingDataGpu(int nFaces, const Face* const* faces, Renderable* renderables);
__global__ void copyNodes(int nNodes, const Node* const* nodes, Vector3f* x, Vector3f* v);
__global__ void copyVertices(int nVertices, const Vertex* const* vertices, Vector2f* u);
__global__ void copyFaces(int nFaces, const Face* const* faces, Pairii* indices);
__global__ void checkEdges(int nEdges, const Edge* const* edges);
__global__ void checkFaces(int nFaces, const Face* const* faces);

template<typename T> __global__ static void findPreservedElements(int n, const T* a, int m, const T* b, int* indices) {
    int nThreads = gridDim.x * blockDim.x;
    int nm = n * m;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nm; i += nThreads) {
        int aIndex = i / m;
        int bIndex = i % m;
        if (a[aIndex] == b[bIndex])
            indices[aIndex] = -1;
    }
}

template<typename T> __global__ static void copyPreservedElements(int nIndices, const int* indices, T* a, T* b) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nIndices; i += nThreads)
        b[i] = a[indices[i]];
}

template<typename T> static void removeGpu(const thrust::device_vector<T>& b, thrust::device_vector<T>& a) {
    int n = a.size(), m = b.size();
    thrust::device_vector<int> indices(n);
    int* indicesPointer = pointer(indices);
    initializeIndices<<<GRID_SIZE, BLOCK_SIZE>>>(n, indicesPointer);
    CUDA_CHECK_LAST();

    findPreservedElements<<<GRID_SIZE, BLOCK_SIZE>>>(n, pointer(a), m, pointer(b), indicesPointer);
    CUDA_CHECK_LAST();

    indices.erase(thrust::remove(indices.begin(), indices.end(), -1), indices.end());
    int nIndices = indices.size();
    thrust::device_vector<T> c(nIndices);
    copyPreservedElements<<<GRID_SIZE, BLOCK_SIZE>>>(nIndices, indicesPointer, pointer(a), pointer(c));
    CUDA_CHECK_LAST();

    a = std::move(c);
}

template<typename T> __global__ static void deleteGpu(int n, const T* const* p) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += nThreads)
        delete p[i];
}

#endif