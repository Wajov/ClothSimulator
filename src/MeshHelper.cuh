#ifndef MESH_HELPER_CUH
#define MESH_HELPER_CUH

#include <device_launch_parameters.h>
#include <thrust/pair.h>

#include "MathHelper.cuh"
#include "Vertex.cuh"
#include "Edge.cuh"
#include "Face.cuh"
#include "Material.cuh"

struct EdgeData {
    Vertex* opposite;
    Face* adjacent;
};

struct VertexData {
    float m;
    Vector3f n;

    __device__ VertexData& operator=(const VertexData& v) {
        m = v.m;
        n = v.n;
        return *this;
    };

    __device__ VertexData operator+(const VertexData& v) const {
        VertexData ans;
        ans.m = m + v.m;
        ans.n = n + v.n;
        return ans;
    };
};

struct IsNull {
    __device__ bool operator()(const Edge* edge) const {
        return edge == nullptr;
    }
};

__global__ static void initializeVertices(int nVertices, const Vertex* vertexArray, Vertex** vertices) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nVertices; i += nThreads) {
        vertices[i] = new Vertex(vertexArray[i].x, vertexArray[i].isFree);
        *vertices[i] = vertexArray[i];
    }
}

__device__ static void setEdgeData(int index0, int index1, const Vertex* vertex, const Face* face, thrust::pair<int, int>& index, EdgeData& edgeData) {
    if (index0 > index1)
        mySwap(index0, index1);
    
    index.first = index0;
    index.second = index1;
    edgeData.opposite = const_cast<Vertex*>(vertex);
    edgeData.adjacent = const_cast<Face*>(face);
}

__global__ static void initializeFaces(int nFaces, const unsigned int* indices, const Vertex* const* vertices, const Material* material, Face** faces, thrust::pair<int, int>* edgeIndices, EdgeData* edgeData) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads) {
        int index0 = 3 * i;
        int index1 = 3 * i + 1;
        int index2 = 3 * i + 2;
        int vertexIndex0 = indices[index0];
        int vertexIndex1 = indices[index1];
        int vertexIndex2 = indices[index2];
        const Vertex* vertex0 = vertices[vertexIndex0];
        const Vertex* vertex1 = vertices[vertexIndex1];
        const Vertex* vertex2 = vertices[vertexIndex2];
        faces[i] = new Face(vertex0, vertex1, vertex2, material);
        setEdgeData(vertexIndex0, vertexIndex1, vertex2, faces[i], edgeIndices[index0], edgeData[index0]);
        setEdgeData(vertexIndex1, vertexIndex2, vertex0, faces[i], edgeIndices[index1], edgeData[index1]);
        setEdgeData(vertexIndex2, vertexIndex0, vertex1, faces[i], edgeIndices[index2], edgeData[index2]);
    }
}

__global__ static void initializeEdges(int nEdges, const thrust::pair<int, int>* indices, const EdgeData* edgeData, const Vertex* const* vertices, Edge** edges) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads)
        if (i == 0 || indices[i] != indices[i - 1]) {
            const thrust::pair<int, int>& index = indices[i];
            const EdgeData& e = edgeData[i];
            edges[i] = new Edge(vertices[index.first], vertices[index.second]);
            edges[i]->setOppositeAndAdjacent(e.opposite, e.adjacent);
            e.adjacent->setEdge(edges[i]);
        } else
            edges[i] = nullptr;
}

__global__ static void setupEdges(int nEdges, const thrust::pair<int, int>* indices, const EdgeData* edgeData, Edge** edges) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads)
        if (i > 0 && indices[i] == indices[i - 1]) {
            const EdgeData& e = edgeData[i];
            edges[i - 1]->setOppositeAndAdjacent(e.opposite, e.adjacent);
        }
}

__global__ static void updateIndicesGpu(int nVertices, Vertex** vertices) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nVertices; i += nThreads)
        vertices[i]->index = i;
}

__global__ static void updateGeometriesVertices(int nVertices, const int* indices, const VertexData* vertexData, Vertex** vertices) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nVertices; i += nThreads) {
        Vertex* vertex = vertices[indices[i]];
        vertex->x1 = vertex->x;
        vertex->m = vertexData[i].m;
        vertex->n = vertexData[i].n.normalized();
    }
}

__global__ static void updateGeometriesEdges(int nEdges, Edge** edges) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads)
        edges[i]->update();
}

__global__ static void updateGeometriesFaces(int nFaces, Face** faces, int* indices, VertexData* vertexData) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads) {
        Face* face = faces[i];
        face->update();
        float m = face->getMass() / 3.0f;
        for (int j = 0; j < 3; j++) {
            Vertex* vertex = face->getVertex(j);
            Vector3f e0 = face->getVertex((j + 1) % 3)->x - vertex->x;
            Vector3f e1 = face->getVertex((j + 2) % 3)->x - vertex->x;

            int index = 3 * i + j;
            indices[index] = vertex->index;
            vertexData[index].m = m;
            vertexData[index].n = e0.cross(e1) / (e0.norm2() * e1.norm2());
        }
    }
}

__global__ static void updateRenderingDataVertices(int nVertices, const Vertex* const* vertices, Vertex* vertexArray) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nVertices; i += nThreads)
        vertexArray[i] = *vertices[i];
}

__global__ static void updateRenderingDataEdges(int nEdges, const Edge* const* edges, unsigned int* indices) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads)
        for (int j = 0; j < 2; j++)
            indices[2 * i + j] = edges[i]->getVertex(j)->index;
}

__global__ static void updateRenderingDataFaces(int nFaces, const Face* const* faces, unsigned int* indices) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads)
        for (int j = 0; j < 3; j++)
            indices[3 * i + j] = faces[i]->getVertex(j)->index;
}

__global__ static void deleteVertices(int nVertices, const Vertex* const* vertices) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nVertices; i += nThreads)
        delete vertices[i];
}

__global__ static void deleteEdges(int nEdges, const Edge* const* edges) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads)
        delete edges[i];
}

__global__ static void deleteFaces(int nFaces, const Face* const* faces) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads)
        delete faces[i];
}

#endif