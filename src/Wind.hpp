#ifndef WIND_HPP
#define WIND_HPP

#include "Vector.hpp"

class Wind {
private:
    float density, drag;
    Vector3f velocity;

public:
    Wind();
    ~Wind();
    float getDensity() const;
    float getDrag() const;
    Vector3f getVelocity() const;
};


#endif