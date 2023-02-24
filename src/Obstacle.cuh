#ifndef OBSTACLE_CUH
#define OBSTACLE_CUH

#include <json/json.h>

#include "Vector.cuh"
#include "Matrix.cuh"
#include "Transform.cuh"
#include "Mesh.cuh"
#include "Shader.cuh"

class Obstacle {
private:
    Mesh* mesh;
    Shader* shader;

public:
    Obstacle(const Json::Value& json);
    ~Obstacle();
    Mesh* getMesh() const;
    void reset() const;
    void bind();
    void render(const Matrix4x4f& model, const Matrix4x4f& view, const Matrix4x4f& projection, const Vector3f& cameraPosition, const Vector3f& lightDirection) const;
};

#endif