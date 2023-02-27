#ifndef SIMULATOR_CUH
#define SIMULATOR_CUH

#include <string>
#include <vector>
#include <functional>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <chrono>

#include <glad/glad.h>
#include <json/json.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>

#include "JsonHelper.cuh"
#include "CudaHelper.cuh"
#include "BVHHelper.cuh"
#include "CollisionHelper.cuh"
#include "SeparationHelper.cuh"
#include "Pair.cuh"
#include "Vector.cuh"
#include "Matrix.cuh"
#include "Magic.cuh"
#include "Wind.cuh"
#include "Cloth.cuh"
#include "Obstacle.cuh"
#include "BVH.cuh"
#include "Impact.cuh"
#include "Intersection.cuh"
#include "optimization/CollisionOptimization.cuh"
#include "optimization/SeparationOptimization.cuh"

extern bool gpu;

class Simulator {
private:
    Magic* magic;
    int frameSteps, nSteps, selectedCloth, selectedFace;
    float frameTime, dt;
    Vector3f gravity;
    Wind* wind;
    std::vector<Cloth*> cloths;
    std::vector<Obstacle*> obstacles;
    unsigned int fbo, indexTexture, rbo;
    Shader* indexShader;
    std::vector<BVH*> buildClothBvhs(bool ccd) const;
    std::vector<BVH*> buildObstacleBvhs(bool ccd) const;
    void updateBvhs(std::vector<BVH*>& bvhs) const;
    void destroyBvhs(const std::vector<BVH*>& bvhs) const;
    void traverse(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, float thickness, std::function<void(const Face*, const Face*, float)> callback) const;
    thrust::device_vector<Proximity> traverse(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, float thickness) const;
    void updateActive(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, const std::vector<Impact>& impacts) const;
    void checkImpacts(const Face* face0, const Face* face1, float thickness, std::vector<Impact>& impacts) const;
    std::vector<Impact> findImpacts(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs) const;
    thrust::device_vector<Impact> findImpactsGpu(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs) const;
    std::vector<Impact> independentImpacts(const std::vector<Impact>& impacts, int deform) const;
    thrust::device_vector<Impact> independentImpacts(const thrust::device_vector<Impact>& impacts, int deform) const;
    void updateActive(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, const std::vector<Intersection>& intersections) const;
    void checkIntersection(const Face* face0, const Face* face1, std::vector<Intersection>& intersections, const std::vector<Cloth*>& cloths, const std::vector<Mesh*>& oldMeshes) const;
    std::vector<Intersection> findIntersections(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, const std::vector<Mesh*>& oldMeshes) const;
    void resetObstacles();
    void physicsStep();
    void collisionStep();
    void remeshingStep();
    void separationStep(const std::vector<Mesh*>& oldMeshes);
    void updateStructures();
    void updateGeometries();
    void updateVelocities();
    void updateRenderingData(bool rebind);

public:
    Simulator(const std::string& path);
    ~Simulator();
    void bind();
    void render(int width, int height, const Matrix4x4f& model, const Matrix4x4f& view, const Matrix4x4f& projection, const Vector3f& cameraPosition, const Vector3f& lightDirection) const;
    void step();
    void printDebugInfo(int x, int y);
};

#endif