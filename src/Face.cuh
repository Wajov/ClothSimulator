#ifndef FACE_CUH
#define FACE_CUH

#include <cuda_runtime.h>

#include "MathHelper.cuh"
#include "Vector.cuh"
#include "Matrix.cuh"
#include "Bounds.cuh"
#include "Node.cuh"
#include "Vertex.cuh"
#include "Edge.cuh"
#include "Material.cuh"

class Edge;

class Face {
public:
    Vertex* vertices[3];
    Edge* edges[3];
    Vector3f n;
    Matrix2x2f inverse;
    float area, mass;
    int minIndex;
    bool removed;
    __host__ __device__ Face(const Vertex* vertex0, const Vertex* vertex1, const Vertex* vertex2, const Material* material);
    __host__ __device__ ~Face();
    __host__ __device__ void initialize(const Material* material);
    __host__ __device__ void setEdge(const Edge* edge);
    __host__ __device__ void setEdges(const Edge* edge0, const Edge* edge1, const Edge* edge2);
    __host__ __device__ void replaceVertex(const Vertex* v, const Vertex* vertex);
    __host__ __device__ void replaceEdge(const Edge* e, const Edge* edge);
    __host__ __device__ bool isFree() const;
    __host__ __device__ bool contain(const Vertex* vertex) const;
    __host__ __device__ bool contain(const Edge* edge) const;
    __host__ __device__ bool adjacent(const Face* face) const;
    __host__ __device__ Edge* findEdge(const Vertex* vertex0, const Vertex* vertex1) const;
    __host__ __device__ Edge* findOpposite(const Node* node) const;
    __host__ __device__ Bounds bounds(bool ccd) const;
    __host__ __device__ Vector3f position(const Vector3f& b) const;
    __host__ __device__ Matrix3x2f derivative(const Vector3f& v0, const Vector3f& v1, const Vector3f& v2) const;
    __host__ __device__ Matrix2x2f curvature() const;
    __host__ __device__ void update();
};

#endif