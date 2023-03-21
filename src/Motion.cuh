#ifndef MOTION_CUH
#define MOTION_CUH

#include <vector>

#include <json/json.h>

#include "JsonHelper.cuh"
#include "Vector.cuh"
#include "Quaternion.cuh"
#include "Transformation.cuh"

class Motion {
private:
    int n, index;
    Transformation a, b, c, d;
    std::vector<float> t;
    std::vector<Transformation> x, v;
    void initialize();
    void updateCoefficients();

public:
    Motion(const Json::Value& json);
    Motion(const std::vector<Vector3f>& translations, const std::vector<Quaternion>& rotations, const Transformation& transformation, float fps);
    ~Motion();
    Transformation computeTransformation(float time);
};

#endif