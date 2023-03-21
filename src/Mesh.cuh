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
#include <thrust/host_vector.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/remove.h>

#include "MathHelper.cuh"
#include "CudaHelper.cuh"
#include "MeshHelper.cuh"
#include "Pair.cuh"
#include "Vector.cuh"
#include "Node.cuh"
#include "Vertex.cuh"
#include "Edge.cuh"
#include "Face.cuh"
#include "Renderable.cuh"
#include "Transformation.cuh"
#include "BackupFace.cuh"
#include "Operator.cuh"

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
    float angle(const Vector3f& x0, const Vector3f& x1, const Vector3f& x2) const;
    void triangulate(const std::vector<Vector3f>& x, const std::vector<int>& xPolygon, const std::vector<int>& uPolygon, std::vector<int>& xTriangles, std::vector<int>& uTriangles) const;
    Edge* findEdge(int index0, int index1, std::unordered_map<Pairii, int, PairHash>& edgeMap);
    void initialize(const std::vector<Vector3f>& x, const std::vector<Vector3f>& v, const std::vector<Vector2f>& u, const std::vector<int>& xIndices, const std::vector<int>& uIndices, const Material* material);

public:
    Mesh(const std::string& path, const Transformation& transformation, const Material* material);
    ~Mesh();
    std::vector<Node*>& getNodes();
    thrust::device_vector<Node*>& getNodesGpu();
    std::vector<Vertex*>& getVertices();
    thrust::device_vector<Vertex*>& getVerticesGpu();
    std::vector<Edge*>& getEdges();
    thrust::device_vector<Edge*>& getEdgesGpu();
    std::vector<Face*>& getFaces();
    thrust::device_vector<Face*>& getFacesGpu();
    bool contain(const Node* node) const;
    bool contain(const Vertex* vertex) const;
    bool contain(const Face* face) const;
    std::vector<BackupFace> backupFaces() const;
    thrust::device_vector<BackupFace> backupFacesGpu() const;
    void apply(const Operator& op);
    void updateIndices();
    void updateNodeGeometries();
    void updateFaceGeometries();
    void updatePositions(float dt);
    void updateVelocities(float dt);
    void updateRenderingData(bool rebind);
    void bind();
    void render() const;
    void load(const std::string& path, const Transformation& transformation, const Material* material);
    void save(const std::string& path);
    void check() const;
};

#endif