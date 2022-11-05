#ifndef MATERIAL_HPP
#define MATERIAL_HPP

#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>

#include <json/json.h>

#include "MathHelper.hpp"
#include "JsonHelper.hpp"
#include "Vector.hpp"
#include "Matrix.hpp"

const int N = 30;

class Material {
private:
    float density, thicken;
    Vector4f stretchingSamples[N][N][N];
    float bendingSamples[3][5];
    Vector4f calculateStretchingSample(const Matrix2x2f& G, const Vector4f (&data)[2][5]) const;

public:
    Material(const Json::Value& json);
    ~Material();
    float getDensity() const;
    float getThicken() const;
    Vector4f stretchingStiffness(const Matrix2x2f& G) const;
    float bendingStiffness(float length, float angle, float area, const Vector2f& d) const;
};

#endif