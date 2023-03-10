#ifndef OPERATOR_CUH
#define OPERATOR_CUH

#include <algorithm>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include <thrust/device_vector.h>
#include <thrust/remove.h>

#include "MathHelper.cuh"
#include "CudaHelper.cuh"
#include "RemeshingHelper.cuh"
#include "Node.cuh"
#include "Vertex.cuh"
#include "Edge.cuh"
#include "Face.cuh"
#include "Material.cuh"

class Operator {
public:
    std::vector<Node*> addedNodes, removedNodes;
    std::vector<Vertex*> addedVertices, removedVertices;
    std::vector<Edge*> addedEdges, removedEdges;
    std::vector<Face*> addedFaces, removedFaces;
    thrust::device_vector<Node*> addedNodesGpu, removedNodesGpu;
    thrust::device_vector<Vertex*> addedVerticesGpu, removedVerticesGpu;
    thrust::device_vector<Edge*> addedEdgesGpu, removedEdgesGpu;
    thrust::device_vector<Face*> addedFacesGpu, removedFacesGpu;
    Operator();
    ~Operator();
    void flip(const Edge* edge, const Material* material);
    void flip(const thrust::device_vector<Edge*>& edges, const Material* material);
    void split(const Edge* edge, const Material* material);
    void split(const thrust::device_vector<Edge*>& edges, const Material* material);
    void collapse(const Edge* edge, int side, const Material* material, const std::unordered_map<Node*, std::vector<Edge*>>& adjacentEdges, const std::unordered_map<Vertex*, std::vector<Face*>>& adjacentFaces);
};

#endif