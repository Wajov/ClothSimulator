#ifndef OBSTACLE_CUH
#define OBSTACLE_CUH

#include <vector>

#include <json/json.h>
#include <thrust/device_vector.h>

#include "CudaHelper.cuh"
#include "ObstacleHelper.cuh"
#include "Vector.cuh"
#include "Matrix.cuh"
#include "Transformation.cuh"
#include "Mesh.cuh"
#include "Motion.cuh"
#include "Shader.cuh"

class Obstacle {
private:
    Mesh* mesh;
    Motion* motion;
    Shader* shader;
    std::vector<Vector3f> base;
    thrust::device_vector<Vector3f> baseGpu;

public:
    Obstacle(const Json::Value& json, const std::vector<Motion*>& motions);
    ~Obstacle();
    Mesh* getMesh() const;
    void transform(float time);
    void bind();
    void render(const Matrix4x4f& model, const Matrix4x4f& view, const Matrix4x4f& projection, const Vector3f& cameraPosition, const Vector3f& lightDirection) const;
};

#endif