#ifndef CLOTH_HPP
#define CLOTH_HPP

#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Cholesky>
#include <json/json.h>

#include "MathHelper.hpp"
#include "Vector.hpp"
#include "Matrix.hpp"
#include "Transform.hpp"
#include "Mesh.hpp"
#include "Material.hpp"
#include "Handle.hpp"
#include "Remeshing.hpp"
#include "Wind.hpp"
#include "Shader.hpp"
#include "BVH.hpp"
#include "NearPoint.hpp"
#include "Plane.hpp"
#include "Disk.hpp"
#include "Operator.hpp"

class Cloth {
private:
    Mesh* mesh;
    Material* material;
    std::vector<Handle*> handles;
    Remeshing* remeshing;
    Shader* edgeShader, * faceShader;
    void addSubMatrix(const Matrix9x9f& B, const Vector3i& indices, Eigen::SparseMatrix<float>& A) const;
    void addSubMatrix(const Matrix12x12f& B, const Vector4i& indices, Eigen::SparseMatrix<float>& A) const;
    void addSubVector(const Vector9f& b, const Vector3i& indices, Eigen::VectorXf& a) const;
    void addSubVector(const Vector12f& b, const Vector4i& indices, Eigen::VectorXf& a) const;
    float distance(const Vector3f& x, const Vector3f& a, const Vector3f& b) const;
    Vector2f barycentricWeights(const Vector3f& x, const Vector3f& a, const Vector3f& b) const;
    std::pair<Vector9f, Matrix9x9f> stretchingForce(const Face* face) const;
    std::pair<Vector12f, Matrix12x12f> bendingForce(const Edge* edge) const;
    void init(Eigen::SparseMatrix<float>& A, Eigen::VectorXf& b) const;
    void addExternalForces(float dt, const Vector3f& gravity, const Wind* wind, Eigen::SparseMatrix<float>& A, Eigen::VectorXf& b) const;
    void addInternalForces(float dt, Eigen::SparseMatrix<float>& A, Eigen::VectorXf& b) const;
    void addHandleForces(float dt, float stiffness, Eigen::SparseMatrix<float>& A, Eigen::VectorXf& b) const;
    Matrix2x2f compressionMetric(const Matrix2x2f& G, const Matrix2x2f& S2) const;
    Matrix2x2f obstacleMetric(const Face* face, const std::vector<Plane>& planes) const;
    Matrix2x2f maxTensor(const Matrix2x2f M[]) const;
    Matrix2x2f faceSizing(const Face* face, const std::vector<Plane>& planes) const;
    float edgeMetric(const Vertex* vertex0, const Vertex* vertex1) const;
    bool shouldFlip(const Edge* edge) const;
    std::vector<Edge*> findEdgesToFlip(const std::vector<Edge*>& edges) const;
    std::vector<Edge*> independentEdges(const std::vector<Edge*>& edges) const;
    bool flipSomeEdges(std::vector<Edge*>& edges, std::vector<Edge*>* edgesToUpdate, std::unordered_map<Vertex*, std::vector<Edge*>>* adjacentEdges, std::unordered_map<Vertex*, std::vector<Face*>>* adjacentFaces) const;
    void flipEdges(std::vector<Edge*>& edges, std::vector<Edge*>* edgesToUpdate, std::unordered_map<Vertex*, std::vector<Edge*>>* adjacentEdges, std::unordered_map<Vertex*, std::vector<Face*>>* adjacentFaces) const;
    std::vector<Edge*> findEdgesToSplit() const;
    bool splitSomeEdges() const;
    void splitEdges();
    bool shouldCollapse(std::unordered_map<Vertex*, std::vector<Edge*>>& adjacentEdges, std::unordered_map<Vertex*, std::vector<Face*>>& adjacentFaces, const Edge* edge, bool reverse) const;
    bool collapseSomeEdges(std::unordered_map<Vertex*, std::vector<Edge*>>& adjacentEdges, std::unordered_map<Vertex*, std::vector<Face*>>& adjacentFaces) const;
    void collapseEdges() const;

public:
    Cloth(const Json::Value& json);
    ~Cloth();
    Mesh* getMesh() const;
    void readDataFromFile(const std::string& path);
    void physicsStep(float dt, float handleStiffness, const Vector3f& gravity, const Wind* wind);
    void remeshingStep(const std::vector<BVH*>& obstacleBvhs, float thickness);
    void updateGeometries();
    void updateVelocities(float dt);
    void updateIndices();
    void updateRenderingData(bool rebind);
    void bind();
    void render(const Matrix4x4f& model, const Matrix4x4f& view, const Matrix4x4f& projection, const Vector3f& cameraPosition, const Vector3f& lightDirection) const;
};

#endif