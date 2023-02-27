#include "JsonHelper.cuh"

int parseInt(const Json::Value& json, int d) {
    return json.isNull() ? d : json.asInt();
}

float parseFloat(const Json::Value& json, float d) {
    return json.isNull() ? d : json.asFloat();
}

std::string parseString(const Json::Value& json, std::string d) {
    return json.isNull() ? d : json.asString();
}

Vector3f parseVector3f(const Json::Value& json, const Vector3f& d) {
    Vector3f ans;
    if (json.isNull())
        return d;
    for (int i = 0; i < 3; i++)
        ans(i) = json[i].asFloat();
    return ans;
}