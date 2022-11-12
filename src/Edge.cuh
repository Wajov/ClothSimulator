#ifndef EDGE_CUH
#define EDGE_CUH

#include <cuda_runtime.h>

#include "Bounds.hpp"
#include "Vertex.cuh"
#include "Face.cuh"

class Face;

class Edge {
private:
    Vertex* vertices[2], * opposites[2];
    Face* adjacents[2];
    float length, angle;

public:
    __host__ __device__ Edge(const Vertex* vertex0, const Vertex* vertex1);
    __host__ __device__ ~Edge();
    __host__ __device__ Vertex* getVertex(int index) const;
    __host__ __device__ void replaceVertex(const Vertex* v, const Vertex* vertex);
    __host__ __device__ Vertex* getOpposite(int index) const;
    __host__ __device__ void replaceOpposite(const Vertex* v, const Vertex* vertex);
    __host__ __device__ Face* getAdjacent(int index) const;
    __host__ __device__ void replaceAdjacent(const Face* f, const Face* face);
    __host__ __device__ void setOppositeAndAdjacent(const Vertex* vertex, const Face* face);
    __host__ __device__ float getLength() const;
    __host__ __device__ float getAngle() const;
    __host__ __device__ bool contain(const Vertex* vertex) const;
    __host__ __device__ bool isBoundary() const;
    __host__ __device__ Bounds bounds(bool ccd) const;
    __host__ __device__ void update();
};

#endif