#ifndef MEMORY_POOL_CUH
#define MEMORY_POOL_CUH

#include <vector>

#include <thrust/device_vector.h>

#include "CudaHelper.cuh"
#include "Vector.cuh"
#include "Node.cuh"
#include "Vertex.cuh"
#include "Edge.cuh"
#include "Face.cuh"

const int NODE_POOL_SIZE = 1000000;
const int VERTEX_POOL_SIZE = 1000000;
const int EDGE_POOL_SIZE = 1000000;
const int FACE_POOL_SIZE = 1000000;

extern bool gpu;

class MemoryPool {
private:
    int nodePointer, vertexPointer, edgePointer, facePointer;
    std::vector<Node> nodePool;
    std::vector<Vertex> vertexPool;
    std::vector<Edge> edgePool;
    std::vector<Face> facePool;
    thrust::device_vector<Node> nodePoolGpu;
    thrust::device_vector<Vertex> vertexPoolGpu;
    thrust::device_vector<Edge> edgePoolGpu;
    thrust::device_vector<Face> facePoolGpu;

public:
    MemoryPool();
    ~MemoryPool();
    Node* createNode(const Vector3f& x, bool isFree);
    Vertex* createVertex(const Vector2f& u);
    Edge* createEdge(const Node* node0, const Node* node1);
    Face* createFace(const Vertex* vertex0, const Vertex* vertex1, const Vertex* vertex2, const Material* material);
    Node* createNodes(int nNodes);
    Vertex* createVertices(int nVertices);
    Edge* createEdges(int nEdges);
    Face* createFaces(int nFaces);
};

#endif