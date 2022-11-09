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

#include "CudaHelper.cuh"
#include "MeshHelper.cuh"
#include "Vector.cuh"
#include "Vertex.cuh"
#include "Edge.cuh"
#include "Face.cuh"
#include "Transform.hpp"
#include "Operator.hpp"

extern bool gpu;

class Mesh {
private:
    std::vector<Vertex*> vertices;
    std::vector<Edge*> edges;
    std::vector<Face*> faces;
    std::vector<Vertex> vertexArray;
    std::vector<unsigned int> edgeIndices, faceIndices;
    unsigned int vbo, edgeVao, edgeEbo, faceVao, faceEbo;
    thrust::device_vector<Vertex*> verticesGpu;
    thrust::device_vector<Edge*> edgesGpu;
    thrust::device_vector<Face*> facesGpu;
    cudaGraphicsResource* vboCuda, * eboCuda;
    void initializeGpu(const Material* material);
    std::vector<std::string> split(const std::string& s, char c) const;
    Edge* findEdge(int index0, int index1, std::map<std::pair<int, int>, int>& edgeMap);

public:
    Mesh(const Json::Value& json, const Transform* transform, const Material* material);
    ~Mesh();
    std::vector<Vertex*>& getVertices();
    std::vector<Edge*>& getEdges();
    std::vector<Face*>& getFaces();
    void readDataFromFile(const std::string& path);
    void apply(const Operator& op);
    void reset();
    void updateGeometries();
    void updateVelocities(float dt);
    void updateIndices();
    void updateRenderingData(bool rebind);
    void bind(const Material* material);
    void renderEdge() const;
    void renderFace() const;
};

#endif