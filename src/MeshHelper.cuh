#ifndef MESH_HELPER_CUH
#define MESH_HELPER_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "MathHelper.cuh"
#include "Vertex.cuh"
#include "Edge.cuh"
#include "Face.cuh"
#include "Material.cuh"

struct EdgeData{
    int index0, index1;
    Vertex* opposite;
    Face* adjacent;
    
    __device__ bool operator==(const EdgeData& e) const {
        return index0 == e.index0 && index1 == e.index1;
    };
};

struct EdgeDataLess {
    __device__ bool operator()(const EdgeData& a, const EdgeData& b) const {
        return a.index0 < b.index0 || a.index0 == b.index0 && a.index1 < b.index1;
    };
};

struct IsNull {
    __device__ bool operator()(const Edge* e) const {
        return e == nullptr;
    };
};

__global__ static void initializeVertices(int nVertices, const Vertex* vertexArray, Vertex** vertices) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int nThreads = gridDim.x * blockDim.x;

    while (index < nVertices) {
        vertices[index] = new Vertex(vertexArray[index].x, vertexArray[index].isFree);
        *vertices[index] = vertexArray[index];
        index += nThreads;
    }
}

__device__ static void setEdgeData(int index0, int index1, const Vertex* vertex, const Face* face, EdgeData& edgeData) {
    if (index0 > index1)
        mySwap(index0, index1);
    
    edgeData.index0 = index0;
    edgeData.index1 = index1;
    edgeData.opposite = const_cast<Vertex*>(vertex);
    edgeData.adjacent = const_cast<Face*>(face);
}

__global__ static void initializeFaces(int nFaces, const unsigned int* indices, const Vertex* const* vertices, const Material* material, Face** faces, EdgeData* edgeData) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int nThreads = gridDim.x * blockDim.x;

    while (index < nFaces) {
        int index0 = 3 * index;
        int index1 = 3 * index + 1;
        int index2 = 3 * index + 2;
        int vertexIndex0 = indices[index0];
        int vertexIndex1 = indices[index1];
        int vertexIndex2 = indices[index2];
        const Vertex* vertex0 = vertices[vertexIndex0];
        const Vertex* vertex1 = vertices[vertexIndex1];
        const Vertex* vertex2 = vertices[vertexIndex2];
        faces[index] = new Face(vertex0, vertex1, vertex2, material);
        setEdgeData(vertexIndex0, vertexIndex1, vertex2, faces[index], edgeData[index0]);
        setEdgeData(vertexIndex1, vertexIndex2, vertex0, faces[index], edgeData[index1]);
        setEdgeData(vertexIndex2, vertexIndex0, vertex1, faces[index], edgeData[index2]);
        index += nThreads;
    }
}

__global__ static void initializeEdges(int nEdges, const EdgeData* edgeData, const Vertex* const* vertices, Edge** edges) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int nThreads = gridDim.x * blockDim.x;

    while (index < nEdges) {
        if (index == 0 || !(edgeData[index] == edgeData[index - 1])) {
            const EdgeData& e = edgeData[index];
            edges[index] = new Edge(vertices[e.index0], vertices[e.index1]);
            edges[index]->setOppositeAndAdjacent(e.opposite, e.adjacent);
            e.adjacent->setEdge(edges[index]);
        } else
            edges[index] = nullptr;
        index += nThreads;
    }
}

__global__ static void setupEdges(int nEdges, const EdgeData* edgeData, Edge** edges) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int nThreads = gridDim.x * blockDim.x;

    while (index < nEdges) {
        if (index > 0 && edgeData[index] == edgeData[index - 1]) {
            const EdgeData& e = edgeData[index];
            edges[index - 1]->setOppositeAndAdjacent(e.opposite, e.adjacent);
        }
        index += nThreads;
    }
}

#endif