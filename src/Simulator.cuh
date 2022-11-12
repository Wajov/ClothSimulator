#ifndef SIMULATOR_CUH
#define SIMULATOR_CUH

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <chrono>

#include <json/json.h>
#include <cuda_runtime.h>

#include "JsonHelper.hpp"
#include "CudaHelper.cuh"
#include "Vector.cuh"
#include "Matrix.cuh"
#include "Magic.hpp"
#include "Wind.cuh"
#include "Cloth.cuh"
#include "Obstacle.hpp"
#include "BVH.hpp"
#include "Impact.hpp"
#include "ImpactZone.hpp"
#include "optimization/ImpactZoneOptimization.hpp"

extern bool gpu;

class Simulator {
private:
    const int MAX_ITERATION;
    Magic* magic;
    int frameSteps, nSteps;
    float frameTime, dt;
    Vector3f gravity;
    Wind* wind, * windGpu;
    std::vector<Cloth*> cloths;
    std::vector<Obstacle*> obstacles;
    void updateActive(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, const std::vector<ImpactZone*>& zones) const;
    void findImpacts(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, std::vector<Impact>& impacts) const;
    std::vector<Impact> independentImpacts(const std::vector<Impact>& impacts) const;
    ImpactZone* findImpactZone(const Vertex* vertex, std::vector<ImpactZone*>& zones) const;
    void addImpacts(const std::vector<Impact>& impacts, std::vector<ImpactZone*>& zones, bool deformObstacles) const;
    void resetObstacles();
    void physicsStep();
    void collisionStep();
    void remeshingStep();
    void updateIndices();
    void updateGeometries();
    void updateVelocities();
    void updateRenderingData(bool rebind);

public:
    Simulator(const std::string& path);
    ~Simulator();
    void bind();
    void render(const Matrix4x4f& model, const Matrix4x4f& view, const Matrix4x4f& projection, const Vector3f& cameraPosition, const Vector3f& lightDirection) const;
    void step();
};

#endif