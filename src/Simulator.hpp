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

const float COLLISION_THICKNESS = 1e-4f;
const int MAXIMUM_ITERATION = 30;

class Simulator {
private:
    float frameTime, frameSteps, dt, obstacleMass;
    Vector3f gravity;
    Wind* wind;
    std::vector<Cloth*> cloths;
    std::vector<Obstacle*> obstacles;
    void physicsStep();
    void findImpacts(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, std::vector<Impact>& impacts) const;
    void updateActive(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, const std::vector<ImpactZone*>& zones) const;
    void collisionStep();

public:
    Simulator(const std::string& path);
    ~Simulator();
    void render(const Matrix4x4f& model, const Matrix4x4f& view, const Matrix4x4f& projection, const Vector3f& cameraPosition, const Vector3f& lightPosition, float lightPower) const;
    void step();
};

#endif