#ifndef CLOTH_HPP
#define CLOTH_HPP

#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Cholesky>
#include <json/json.h>

#include "TypeHelper.hpp"
#include "MathHelper.hpp"
#include "Mesh.hpp"
#include "Material.hpp"
#include "Wind.hpp"
#include "Handle.hpp"
#include "constraint/Constraint.hpp"

class Cloth {
private:
    Mesh* mesh;
    std::vector<Material*> materials;
    std::vector<Handle*> handles;
    Vector3i indices(const Vertex* v0, const Vertex* v1, const Vertex* v2) const;
    Vector4i indices(const Vertex* v0, const Vertex* v1, const Vertex* v2, const Vertex* v3) const;
    void addSubMatrix(const MatrixXxXf& B, const VectorXi& indices, Eigen::SparseMatrix<float>& A) const;
    void addSubVector(const VectorXf& b, const VectorXi& indices, VectorXf& a) const;
    float distance(const Vector3f& x, const Vector3f& a, const Vector3f& b) const;
    Vector2f barycentricWeights(const Vector3f& x, const Vector3f& a, const Vector3f& b) const;
    std::pair<Vector9f, Matrix9x9f> stretchingForce(const Face* face, const Material* material) const;
    std::pair<Vector12f, Matrix12x12f> bendingForce(const Edge* edge, const Material* material) const;
    std::vector<Constraint*> getConstraints() const;
    void init(Eigen::SparseMatrix<float>& A, VectorXf& b) const;
    void addExternalForces(float dt, const Vector3f& gravity, const Wind* wind, Eigen::SparseMatrix<float>& A, VectorXf& b) const;
    void addInternalForces(float dt, const Material* material, Eigen::SparseMatrix<float>& A, VectorXf& b) const;
    void addConstraintForces(float dt, const std::vector<Constraint*>& constraints, Eigen::SparseMatrix<float>& A, VectorXf& b) const;

public:
    Cloth(const Json::Value& json);
    ~Cloth();
    Mesh* getMesh() const;
    void addHandle(int index);
    void readDataFromFile(const std::string& path);
    void renderEdge() const;
    void renderFace() const;
    void update(float dt, const Vector3f& gravity, const Wind* wind);
};

#endif