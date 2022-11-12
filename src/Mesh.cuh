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
    unsigned int vbo, edgeVao, edgeEbo, faceVao, faceEbo;
    thrust::device_vector<Vertex*> verticesGpu;
    thrust::device_vector<Edge*> edgesGpu;
    thrust::device_vector<Face*> facesGpu;
    cudaGraphicsResource* vboCuda, * edgeEboCuda, * faceEboCuda;
    std::vector<std::string> split(const std::string& s, char c) const;
    Edge* findEdge(int index0, int index1, std::map<std::pair<int, int>, int>& edgeMap);
    void initialize(const std::vector<Vertex>& vertexArray, const std::vector<unsigned int>& indices, const Material* material);

public:
    Mesh(const Json::Value& json, const Transform* transform, const Material* material);
    ~Mesh();
    std::vector<Vertex*>& getVertices();
    thrust::device_vector<Vertex*>& getVerticesGpu();
    std::vector<Edge*>& getEdges();
    thrust::device_vector<Edge*>& getEdgesGpu();
    std::vector<Face*>& getFaces();
    thrust::device_vector<Face*>& getFacesGpu();
    void readDataFromFile(const std::string& path);
    void apply(const Operator& op);
    void reset();
    void updateIndices();
    void updateGeometries();
    void updateVelocities(float dt);
    void updateRenderingData(bool rebind);
    void bind(const Material* material);
    void renderEdges() const;
    void renderFaces() const;
};

#endif