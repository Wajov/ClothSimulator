#ifndef EDGE_CUH
#define EDGE_CUH

#include <cuda_runtime.h>

#include "Bounds.hpp"
#include "Node.cuh"
#include "Vertex.cuh"
#include "Face.cuh"

class Face;

class Edge {
public:
    Node* nodes[2];
    Vertex* vertices[2][2], * opposites[2];
    Face* adjacents[2];
    float length, angle;
    __host__ __device__ Edge(const Node* node0, const Node* node1);
    __host__ __device__ ~Edge();
    __host__ __device__ void initialize(const Vertex* vertex, const Face* face);
    __host__ __device__ void replaceNode(const Node* n, const Node* node);
    __host__ __device__ void replaceVertex(const Vertex* v, const Vertex* vertex);
    __host__ __device__ void replaceOpposite(const Vertex* v, const Vertex* vertex);
    __host__ __device__ void replaceAdjacent(const Face* f, const Face* face);
    __host__ __device__ bool isBoundary() const;
    __host__ __device__ bool isSeam() const;
    __host__ __device__ Bounds bounds(bool ccd) const;
    __host__ __device__ void update();
};

#endif