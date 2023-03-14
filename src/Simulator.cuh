#ifndef SIMULATOR_CUH
#define SIMULATOR_CUH

#include <string>
#include <vector>
#include <functional>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <chrono>
#include <filesystem>

#include <glad/glad.h>
#include <json/json.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/reduce.h>

#include "JsonHelper.cuh"
#include "CudaHelper.cuh"
#include "BVHHelper.cuh"
#include "CollisionHelper.cuh"
#include "SeparationHelper.cuh"
#include "Renderer.cuh"
#include "Pair.cuh"
#include "Vector.cuh"
#include "Matrix.cuh"
#include "Magic.cuh"
#include "Wind.cuh"
#include "Motion.cuh"
#include "Cloth.cuh"
#include "Obstacle.cuh"
#include "BVH.cuh"
#include "Impact.cuh"
#include "BackupFace.cuh"
#include "Intersection.cuh"
#include "optimization/CollisionOptimization.cuh"
#include "optimization/SeparationOptimization.cuh"

extern bool gpu;

enum SimulationMode {
    Simulate,
    SimulateOffline,
    Resume,
    ResumeOffline,
    Replay
};

class Simulator {
private:
    SimulationMode mode;
    Json::Value json;
    std::string directory;
    Magic* magic;
    int frameSteps, endFrame, nSteps, nFrames;
    float frameTime, endTime, dt;
    Vector3f gravity;
    Wind* wind;
    std::vector<Motion*> motions;
    std::vector<Cloth*> cloths;
    std::vector<Obstacle*> obstacles;
    Renderer* renderer;
    std::vector<BVH*> buildClothBvhs(bool ccd) const;
    std::vector<BVH*> buildObstacleBvhs(bool ccd) const;
    void updateBvhs(std::vector<BVH*>& bvhs) const;
    void destroyBvhs(const std::vector<BVH*>& bvhs) const;
    void traverse(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, float thickness, std::function<void(const Face*, const Face*, float)> callback) const;
    thrust::device_vector<Proximity> traverse(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, float thickness) const;
    void updateActive(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, const std::vector<Impact>& impacts) const;
    void checkImpacts(const Face* face0, const Face* face1, float thickness, std::vector<Impact>& impacts) const;
    thrust::device_vector<Impact> findImpacts(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs) const;
    std::vector<Impact> independentImpacts(const std::vector<Impact>& impacts, int deform) const;
    thrust::device_vector<Impact> independentImpacts(const thrust::device_vector<Impact>& impacts, int deform) const;
    void updateActive(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, const std::vector<Intersection>& intersections) const;
    Vector3f oldPosition(const Vector2f& u, const std::vector<BackupFace>& faces) const;
    Vector3f oldPosition(const Face* face, const Vector3f& b, const std::vector<std::vector<BackupFace>>& faces) const;
    void checkIntersection(const Face* face0, const Face* face1, std::vector<Intersection>& intersections, const std::vector<std::vector<BackupFace>>& faces) const;
    thrust::device_vector<Intersection> findIntersections(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, const std::vector<thrust::device_vector<BackupFace>>& faces) const;
    void obstacleStep();
    void physicsStep();
    void collisionStep();
    void remeshingStep();
    void separationStep(const std::vector<std::vector<BackupFace>>& faces);
    void separationStep(const std::vector<thrust::device_vector<BackupFace>>& faces);
    void updateStructures();
    void updateClothNodeGeometries();
    void updateObstacleNodeGeometries();
    void updateClothFaceGeometries();
    void updateObstacleFaceGeometries();
    void updateVelocities();
    void updateRenderingData(bool rebind);
    void simulateStep(bool offline);
    void replayStep();
    void bind();
    void render() const;
    bool load();
    void save();
    int lastFrame() const;
    bool finished() const;
    void simulate();
    void simulateOffline();
    void resume();
    void resumeOffline();
    void replay();

public:
    Simulator(SimulationMode mode, const std::string& path, const std::string& directory);
    ~Simulator();
    void start();
};

#endif