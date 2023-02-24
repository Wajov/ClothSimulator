#ifndef JSON_HELPER_CUH
#define JSON_HELPER_CUH

#include <json/json.h>

#include "Vector.cuh"

static int parseInt(const Json::Value& json, int d = 0) {
    return json.isNull() ? d : json.asInt();
}

static float parseFloat(const Json::Value& json, float d = 0.0f) {
    return json.isNull() ? d : json.asFloat();
}

static std::string parseString(const Json::Value& json, std::string d = "") {
    return json.isNull() ? d : json.asString();
}

static Vector3f parseVector3f(const Json::Value& json, const Vector3f& d = Vector3f(0.0f, 0.0f, 0.0f)) {
    Vector3f ans;
    if (json.isNull())
        return d;
    for (int i = 0; i < 3; i++)
        ans(i) = json[i].asFloat();
    return ans;
}

#endif