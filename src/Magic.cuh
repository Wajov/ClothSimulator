#ifndef MAGIC_CUH
#define MAGIC_CUH

#include <json/json.h>

#include "JsonHelper.cuh"

class Magic {
public:
    float handleStiffness, collisionStiffness;
    float repulsionThickness, collisionThickness;
    Magic(const Json::Value& json);
    ~Magic();
};

#endif