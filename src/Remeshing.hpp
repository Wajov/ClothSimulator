#ifndef REMESHING_HPP
#define REMESHING_HPP

#include <cmath>

#include <json/json.h>

#include "JsonHelper.hpp"

class Remeshing {
public:
    float refineAngle, refineVelocity, refineCompression, ribStiffening;
    float sizeMin, sizeMax;
    float aspectMin;
    float flipThreshold;
    Remeshing(const Json::Value& json);
    ~Remeshing();
};

#endif