#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include <json/json.h>

#include "TypeHelper.hpp"
#include "JsonHelper.hpp"
#include "Wind.hpp"
#include "Cloth.hpp"
#include "Obstacle.hpp"
#include "BVH.hpp"
#include "Impact.hpp"
#include "ImpactZone.hpp"
#include "optimization/ImpactZoneOptimization.hpp"

class Simulator {
private:
    const float COLLISION_THICKNESS;
    const int MAX_ITERATION;
    float frameTime, frameSteps, dt;
    Vector3f gravity;
    Wind* wind;
    std::vector<Cloth*> cloths;
    std::vector<Obstacle*> obstacles;
    void updateActive(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, const std::vector<ImpactZone*>& zones) const;
    void findImpacts(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, std::vector<Impact>& impacts) const;
    std::vector<Impact> independentImpacts(const std::vector<Impact>& impacts) const;
    ImpactZone* findImpactZone(const Vertex* vertex, std::vector<ImpactZone*>& zones) const;
    void addImpacts(const std::vector<Impact>& impacts, std::vector<ImpactZone*>& zones, bool deformObstacles) const;
    void physicsStep();
    void collisionStep();

public:
    Simulator(const std::string& path);
    ~Simulator();
    void render(const Matrix4x4f& model, const Matrix4x4f& view, const Matrix4x4f& projection, const Vector3f& cameraPosition, const Vector3f& lightPosition, float lightPower) const;
    void step();
};

#endif