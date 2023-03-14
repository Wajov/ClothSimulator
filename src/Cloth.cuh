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
#include "RemeshingHelper.cuh"
#include "Pair.cuh"
#include "Vector.cuh"
#include "Matrix.cuh"
#include "Transformation.cuh"
#include "Mesh.cuh"
#include "Material.cuh"
#include "Handle.cuh"
#include "Remeshing.cuh"
#include "Wind.cuh"
#include "Shader.cuh"
#include "BVH.cuh"
#include "NearPoint.cuh"
#include "Plane.cuh"
#include "Disk.cuh"
#include "Operator.cuh"

extern bool gpu;

class Cloth {
private:
    Mesh* mesh;
    Material* material;
    std::vector<Handle> handles;
    thrust::device_vector<Handle> handlesGpu;
    Remeshing* remeshing;
    Shader* edgeShader, * faceShader;
    cusparseHandle_t cusparseHandle;
    cusolverSpHandle_t cusolverHandle;
    void addSubMatrix(const Matrix9x9f& B, const Vector3i& indices, Eigen::SparseMatrix<float>& A) const;
    void addSubMatrix(const Matrix12x12f& B, const Vector4i& indices, Eigen::SparseMatrix<float>& A) const;
    void addSubVector(const Vector9f& b, const Vector3i& indices, Eigen::VectorXf& a) const;
    void addSubVector(const Vector12f& b, const Vector4i& indices, Eigen::VectorXf& a) const;
    void initializeForces(Eigen::SparseMatrix<float>& A, Eigen::VectorXf& b) const;
    void addExternalForces(float dt, const Vector3f& gravity, const Wind* wind, Eigen::SparseMatrix<float>& A, Eigen::VectorXf& b) const;
    void addInternalForces(float dt, Eigen::SparseMatrix<float>& A, Eigen::VectorXf& b) const;
    void addHandleForces(float dt, float stiffness, Eigen::SparseMatrix<float>& A, Eigen::VectorXf& b) const;
    std::vector<Plane> findNearestPlane(const std::vector<BVH*>& obstacleBvhs, float thickness) const;
    thrust::device_vector<Plane> findNearestPlaneGpu(const std::vector<BVH*>& obstacleBvhs, float thickness) const;
    void computeSizing(const std::vector<Plane>& planes);
    void computeSizing(const thrust::device_vector<Plane>& planes);
    std::vector<Edge*> findEdgesToFlip() const;
    thrust::device_vector<Edge*> findEdgesToFlipGpu() const;
    bool flipSomeEdges();
    void flipEdges();
    std::vector<Edge*> findEdgesToSplit() const;
    thrust::device_vector<Edge*> findEdgesToSplitGpu() const;
    bool splitSomeEdges();
    void splitEdges();
    void buildAdjacents(std::unordered_map<Node*, std::vector<Edge*>>& adjacentEdges, std::unordered_map<Vertex*, std::vector<Face*>>& adjacentFaces) const;
    void buildAdjacents(thrust::device_vector<int>& edgeBegin, thrust::device_vector<int>& edgeEnd, thrust::device_vector<Edge*>& adjacentEdges, thrust::device_vector<int>& faceBegin, thrust::device_vector<int>& faceEnd, thrust::device_vector<Face*>& adjacentFaces) const;
    bool shouldCollapse(const Edge* edge, int side, const std::unordered_map<Node*, std::vector<Edge*>>& adjacentEdges, const std::unordered_map<Vertex*, std::vector<Face*>>& adjacentFaces) const;
    std::vector<Pairei> findEdgesToCollapse(const std::unordered_map<Node*, std::vector<Edge*>>& adjacentEdges, const std::unordered_map<Vertex*, std::vector<Face*>>& adjacentFaces) const;
    thrust::device_vector<Pairei> findEdgesToCollapse(const thrust::device_vector<int>& edgeBegin, const thrust::device_vector<int>& edgeEnd, const thrust::device_vector<Edge*>& adjacentEdges, const thrust::device_vector<int>& faceBegin, const thrust::device_vector<int>& faceEnd, const thrust::device_vector<Face*>& adjacentFaces) const;
    bool collapseSomeEdges();
    void collapseEdges();

public:
    Cloth(const Json::Value& json);
    ~Cloth();
    Mesh* getMesh() const;
    void physicsStep(float dt, float handleStiffness, const Vector3f& gravity, const Wind* wind);
    void remeshingStep(const std::vector<BVH*>& obstacleBvhs, float thickness);
    void bind();
    void render(const Matrix4x4f& model, const Matrix4x4f& view, const Matrix4x4f& projection, const Vector3f& cameraPosition, const Vector3f& lightDirection) const;
    void load(const std::string& path);
    void save(const std::string& path, Json::Value& json) const;
};

#endif