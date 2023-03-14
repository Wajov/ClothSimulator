#ifndef JSON_HELPER_CUH
#define JSON_HELPER_CUH

#include <json/json.h>

#include "Vector.cuh"

int parseInt(const Json::Value& json, int d = 0);
float parseFloat(const Json::Value& json, float d = 0.0f);
std::string parseString(const Json::Value& json, std::string d = "");
Vector3f parseVector3f(const Json::Value& json, const Vector3f& d = Vector3f());
Vector4f parseVector4f(const Json::Value& json, const Vector4f& d = Vector4f());

#endif