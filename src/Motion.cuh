#ifndef MOTION_CUH
#define MOTION_CUH

#include <json/json.h>

#include "JsonHelper.cuh"
#include "Transformation.cuh"

class Motion {
private:
    int n, index;
    Transformation a, b, c, d;
    std::vector<float> t;
    std::vector<Transformation> x, v;
    void updateCoefficients();

public:
    Motion(const Json::Value& json);
    ~Motion();
    Transformation computeTransformation(float time);
};

#endif