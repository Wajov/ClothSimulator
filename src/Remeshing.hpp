#ifndef REMESHING_HPP
#define REMESHING_HPP

#include <cmath>

#include <json/json.h>

#include "JsonHelper.hpp"

class Remeshing {
public:
    float refineAngle, refineVelocity, refineCompression;
    float sizeMin, sizeMax;
    float aspectMin;
    Remeshing(const Json::Value& json);
    ~Remeshing();
};

#endif