#include "Wind.cuh"

Wind::Wind(const Json::Value& json) {
    density = parseFloat(json["density"], 1.0f);
    drag = parseFloat(json["drag"]);
    velocity = parseVector3f(json["velocity"]);
}

Wind::~Wind() {}

float Wind::getDensity() const {
    return density;
}

float Wind::getDrag() const {
    return drag;
}

Vector3f Wind::getVelocity() const {
    return velocity;
}
