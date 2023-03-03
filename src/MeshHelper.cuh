#ifndef MESH_HELPER_CUH
#define MESH_HELPER_CUH

#include <device_launch_parameters.h>

#include "CudaHelper.cuh"
#include "MathHelper.cuh"
#include "Pair.cuh"
#include "Node.cuh"
#include "Vertex.cuh"
#include "Edge.cuh"
#include "Face.cuh"
#include "Renderable.cuh"
#include "Material.cuh"

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

__global__ void initializeNodes(int nNodes, const Vector3f* x, bool isFree, Node** nodes);
__global__ void initializeVertices(int nVertices, const Vector2f* u, Vertex** vertices);
__device__ void setEdgeData(int index0, int index1, const Vertex* vertex, const Face* face, Pairii& index, EdgeData& edgeData);
__global__ void initializeFaces(int nFaces, const int* xIndices, const int* uIndices, const Node* const* nodes, const Material* material, Vertex** vertices, Face** faces, Pairii* edgeIndices, EdgeData* edgeData);
__global__ void initializeEdges(int nEdges, const Pairii* indices, const EdgeData* edgeData, const Node* const* nodes, Edge** edges);
__global__ void setEdges(int nEdges, const Pairii* indices, const EdgeData* edgeData, Edge** edges);
__global__ void collectPreservedNodes(int nEdges, const Edge* const* edges, int* nodeIndices);
__global__ void setPreservedNodes(int nIndices, const int* indices, Node** nodes);
__global__ void resetGpu(int nNodes, Node** nodes);
__global__ void updateNodeIndices(int nNodes, Node** nodes);
__global__ void updateVertexIndices(int nVertices, Vertex** vertices);
__global__ void collectNodeStructures(int nFaces, Face** faces, int* indices, NodeData* nodeData);
__global__ void setNodeStructures(int nIndices, const int* indices, const NodeData* nodeData, Node** nodes);
__global__ void updateNodeGeometries(int nNodes, Node** nodes);
__global__ void updateEdgeGeometries(int nEdges, Edge** edges);
__global__ void updateFaceGeometries(int nFaces, Face** faces);
__global__ void collectNodeGeometries(int nFaces, Face** faces, int* indices, Vector3f* nodeData);
__global__ void setNodeGeometries(int nIndices, const int* indices, const Vector3f* nodeData, Node** nodes);
__global__ void updateVelocitiesGpu(int nNodes, float invDt, Node** nodes);
__global__ void updateRenderingDataGpu(int nFaces, const Face* const* faces, Renderable* renderables);
__global__ void copyX(int nNodes, const Node* const* nodes, Vector3f* x);
__global__ void copyV(int nNodes, const Node* const* nodes, Vector3f* v);
__global__ void copyU(int nVertices, const Vertex* const* vertices, Vector2f* u);
__global__ void copyFaceIndices(int nFaces, const Face* const* faces, Pairii* indices);
__global__ void printDebugInfoGpu(const Face* const* faces, int index);
__global__ void deleteNodes(int nNodes, const Node* const* nodes);
__global__ void deleteVertices(int nVertices, const Vertex* const* vertices);
__global__ void deleteEdges(int nEdges, const Edge* const* edges);
__global__ void deleteFaces(int nFaces, const Face* const* faces);

#endif