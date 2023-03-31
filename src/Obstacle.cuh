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
#include "Shader.cuh"

class Obstacle {
private:
    int motionIndex;
    Mesh* mesh;
    Shader* shader;
    std::vector<Vector3f> base;
    thrust::device_vector<Vector3f> baseGpu;
    void initialize();

public:
    Obstacle(const Json::Value& json, const std::vector<Transformation>& transformations, MemoryPool* pool);
    Obstacle(const std::string& path, int motionIndex, const std::vector<Transformation>& transformations, MemoryPool* pool);
    ~Obstacle();
    Mesh* getMesh() const;
    void transform(const std::vector<Transformation>& transformations);
    void step(float dt, const std::vector<Transformation>& transformations);
    void bind();
    void render(const Matrix4x4f& model, const Matrix4x4f& view, const Matrix4x4f& projection, const Vector3f& cameraPosition, const Vector3f& lightDirection) const;
};

#endif