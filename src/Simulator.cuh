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

#include "JsonHelper.cuh"
#include "CudaHelper.cuh"
#include "CollisionHelper.cuh"
#include "SeparationHelper.cuh"
#include "Vector.cuh"
#include "Matrix.cuh"
#include "Magic.cuh"
#include "Wind.cuh"
#include "Cloth.cuh"
#include "Obstacle.cuh"
#include "BVH.cuh"
#include "Impact.cuh"
#include "ImpactZone.cuh"
#include "Intersection.cuh"
#include "optimization/ImpactZoneOptimization.cuh"
#include "optimization/SeparationOptimization.cuh"

extern bool gpu;

struct Pixel {
    int clothIndex, faceInedx;
};

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
    std::vector<Impact> independentImpacts(const std::vector<Impact>& impacts) const;
    ImpactZone* findImpactZone(const Node* node, std::vector<ImpactZone*>& zones) const;
    void addImpacts(const std::vector<Impact>& impacts, std::vector<ImpactZone*>& zones, bool deformObstacles) const;
    void updateActive(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, const std::vector<ImpactZone*>& zones) const;
    void updateActive(const std::vector<BVH*>& clothBvhs, const std::vector<Intersection>& intersections) const;
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