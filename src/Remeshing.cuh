#ifndef REMESHING_CUH
#define REMESHING_CUH

#include <cmath>

#include <json/json.h>

#include "JsonHelper.cuh"

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