#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include <json/json.h>

#include "TypeHelper.hpp"
#include "Cloth.hpp"
#include "Wind.hpp"

class Simulator {
private:
    float dt;
    Vector3f gravity;
    Wind* wind;
    std::vector<Cloth*> cloths;

public:
    Simulator(const std::string& path);
    ~Simulator();
    void renderEdge() const;
    void renderFace() const;
    void update();
};

#endif