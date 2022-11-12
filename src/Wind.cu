#include "Wind.cuh"

Wind::Wind() : 
    density(1.0f),
    drag(0.0f),
    velocity(0.0f, 0.0f, 0.0f) {}

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
