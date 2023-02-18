#ifndef MESH_CUH
#define MESH_CUH

#include <vector>
#include <map>
#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <json/json.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/reduce.h>

#include "MathHelper.cuh"
#include "CudaHelper.cuh"
#include "MeshHelper.cuh"
#include "Vector.cuh"
#include "Node.cuh"
#include "Vertex.cuh"
#include "Edge.cuh"
#include "Face.cuh"
#include "Renderable.cuh"
#include "Transform.hpp"
#include "Operator.hpp"

extern bool gpu;

class Mesh {
private:
    std::vector<Node*> nodes;
    std::vector<Vertex*> vertices;
    std::vector<Edge*> edges;
    std::vector<Face*> faces;
    unsigned int vao, vbo;
    thrust::device_vector<Node*> nodesGpu;
    thrust::device_vector<Vertex*> verticesGpu;
    thrust::device_vector<Edge*> edgesGpu;
    thrust::device_vector<Face*> facesGpu;
    cudaGraphicsResource* vboCuda;
    std::vector<std::string> split(const std::string& s, char c) const;
    Edge* findEdge(int index0, int index1, std::map<std::pair<int, int>, int>& edgeMap);
    void initialize(const std::vector<Vector3f>& x, const std::vector<Vector2f>& u, const std::vector<int>& xIndices, const std::vector<int>& uIndices, const Material* material);

public:
    Mesh(const Json::Value& json, const Transform* transform, const Material* material);
    Mesh(const Mesh* mesh);
    ~Mesh();
    std::vector<Node*>& getNodes();
    thrust::device_vector<Node*>& getNodesGpu();
    std::vector<Vertex*>& getVertices();
    thrust::device_vector<Vertex*>& getVerticesGpu();
    std::vector<Edge*>& getEdges();
    thrust::device_vector<Edge*>& getEdgesGpu();
    std::vector<Face*>& getFaces();
    thrust::device_vector<Face*>& getFacesGpu();
    bool contain(const Vertex* vertex) const;
    bool contain(const Face* face) const;
    void reset();
    Vector3f oldPosition(const Vector2f& u) const;
    void apply(const Operator& op);
    void updateStructures();
    void updateGeometries(float dt);
    void updateRenderingData(bool rebind);
    void bind();
    void render() const;
    void readDataFromFile(const std::string& path);
    void writeDataToFile(const std::string& path);
    void printDebugInfo(int selectedFace);
};

#endif