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

class Simulator {
private:
    float frameTime, frameSteps, dt;
    Vector3f gravity;
    Wind* wind;
    std::vector<Cloth*> cloths;
    std::vector<Obstacle*> obstacles;
    void physicsStep();

public:
    Simulator(const std::string& path);
    ~Simulator();
    void render(const Matrix4x4f& model, const Matrix4x4f& view, const Matrix4x4f& projection, const Vector3f& cameraPosition, const Vector3f& lightPosition, float lightPower) const;
    void step();
};

#endif