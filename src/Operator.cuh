#ifndef OPERATOR_CUH
#define OPERATOR_CUH

#include <algorithm>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include "MathHelper.cuh"
#include "Node.cuh"
#include "Vertex.cuh"
#include "Edge.cuh"
#include "Face.cuh"
#include "Material.cuh"

class Operator {
private:
    void updateActive(const std::vector<Face*>& activeFaces);

public:
    std::vector<Node*> addedNodes, removedNodes;
    std::vector<Vertex*> addedVertices, removedVertices;
    std::vector<Edge*> addedEdges, removedEdges;
    std::vector<Face*> addedFaces, removedFaces, activeFaces;
    Operator();
    ~Operator();
    void flip(const Edge* edge, const Material* material);
    void split(const Edge* edge, const Material* material, int index);
    void collapse(const Edge* edge, int side, const Material* material, std::unordered_map<Node*, std::vector<Edge*>>& adjacentEdges, std::unordered_map<Vertex*, std::vector<Face*>>& adjacentFaces);
};

#endif