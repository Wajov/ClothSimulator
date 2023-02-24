#ifndef CLOTH_CUH
#define CLOTH_CUH

#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <chrono>

#include <json/json.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <cusparse.h>
#include <cusolverSp.h>

#include "MathHelper.cuh"
#include "CudaHelper.cuh"
#include "ClothHelper.cuh"
#include "PhysicsHelper.cuh"
#include "Vector.cuh"
#include "Matrix.cuh"
#include "Transform.hpp"
#include "Mesh.cuh"
#include "Material.cuh"
#include "Handle.cuh"
#include "Remeshing.hpp"
#include "Wind.cuh"
#include "Shader.hpp"
#include "BVH.cuh"
#include "NearPoint.hpp"
#include "Plane.hpp"
#include "Disk.hpp"
#include "Operator.hpp"

extern bool gpu;

class Cloth {
private:
    Mesh* mesh;
    Material* material;
    std::vector<Handle*> handles;
    thrust::device_vector<Handle*> handlesGpu;
    Remeshing* remeshing;
    Shader* edgeShader, * faceShader;
    cusparseHandle_t cusparseHandle;
    cusolverSpHandle_t cusolverHandle;
    void addSubMatrix(const Matrix9x9f& B, const Vector3i& indices, Eigen::SparseMatrix<float>& A) const;
    void addSubMatrix(const Matrix12x12f& B, const Vector4i& indices, Eigen::SparseMatrix<float>& A) const;
    void addSubVector(const Vector9f& b, const Vector3i& indices, Eigen::VectorXf& a) const;
    void addSubVector(const Vector12f& b, const Vector4i& indices, Eigen::VectorXf& a) const;
    void init(Eigen::SparseMatrix<float>& A, Eigen::VectorXf& b) const;
    void addExternalForces(float dt, const Vector3f& gravity, const Wind* wind, Eigen::SparseMatrix<float>& A, Eigen::VectorXf& b) const;
    void addInternalForces(float dt, Eigen::SparseMatrix<float>& A, Eigen::VectorXf& b) const;
    void addHandleForces(float dt, float stiffness, Eigen::SparseMatrix<float>& A, Eigen::VectorXf& b) const;
    Matrix2x2f compressionMetric(const Matrix2x2f& G, const Matrix2x2f& S2) const;
    Matrix2x2f obstacleMetric(const Face* face, const std::vector<Plane>& planes) const;
    Matrix2x2f maxTensor(const Matrix2x2f M[]) const;
    Matrix2x2f faceSizing(const Face* face, const std::vector<Plane>& planes) const;
    float edgeMetric(const Vertex* vertex0, const Vertex* vertex1) const;
    float edgeMetric(const Edge* edge) const;
    bool shouldFlip(const Edge* edge) const;
    std::vector<Edge*> findEdgesToFlip(const std::vector<Edge*>& edges) const;
    std::vector<Edge*> independentEdges(const std::vector<Edge*>& edges) const;
    bool flipSomeEdges(std::vector<Edge*>& edges, std::vector<Edge*>* edgesToUpdate, std::unordered_map<Node*, std::vector<Edge*>>* adjacentEdges, std::unordered_map<Vertex*, std::vector<Face*>>* adjacentFaces) const;
    void flipEdges(std::vector<Edge*>& edges, std::vector<Edge*>* edgesToUpdate, std::unordered_map<Node*, std::vector<Edge*>>* adjacentEdges, std::unordered_map<Vertex*, std::vector<Face*>>* adjacentFaces) const;
    std::vector<Edge*> findEdgesToSplit() const;
    bool splitSomeEdges() const;
    void splitEdges();
    bool shouldCollapse(std::unordered_map<Node*, std::vector<Edge*>>& adjacentEdges, std::unordered_map<Vertex*, std::vector<Face*>>& adjacentFaces, const Edge* edge, int side) const;
    bool collapseSomeEdges(std::unordered_map<Node*, std::vector<Edge*>>& adjacentEdges, std::unordered_map<Vertex*, std::vector<Face*>>& adjacentFaces) const;
    void collapseEdges() const;

public:
    Cloth(const Json::Value& json);
    ~Cloth();
    Mesh* getMesh() const;
    void readDataFromFile(const std::string& path);
    void physicsStep(float dt, float handleStiffness, const Vector3f& gravity, const Wind* wind);
    void remeshingStep(const std::vector<BVH*>& obstacleBvhs, float thickness);
    void updateStructures();
    void updateGeometries();
    void updateVelocities(float dt);
    void updateRenderingData(bool rebind);
    void bind();
    void render(const Matrix4x4f& model, const Matrix4x4f& view, const Matrix4x4f& projection, const Vector3f& cameraPosition, const Vector3f& lightDirection, int selectedFace) const;
    void printDebugInfo(int selectedFace);
};

#endif