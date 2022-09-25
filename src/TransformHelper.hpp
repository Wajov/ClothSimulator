#ifndef TRANSFORM_HELPER_HPP
#define TRANSFORM_HELPER_HPP

#include <cmath>

#include "TypeHelper.hpp"

static Matrix4x4f scale(const float scaling) {
    Matrix4x4f ans = Matrix4x4f::Zero();

    ans(0, 0) = ans(1, 1) = ans(2, 2) = scaling;
    ans(3, 3) = 1.0f;

    return ans;
}

static Matrix4x4f rotate(const Vector3f& v, const float angle) {
    Vector3f axis = v.normalized();
    float s = std::sin(angle);
    float c = std::cos(angle);
    Matrix4x4f ans = Matrix4x4f::Zero();

    ans(0, 0) = (1.0f - c) * axis(0) * axis(0) + c;
    ans(0, 1) = (1.0f - c) * axis(1) * axis(0) - s * axis(2);
    ans(0, 2) = (1.0f - c) * axis(2) * axis(0) + s * axis(1);

    ans(1, 0) = (1.0f - c) * axis(0) * axis(1) + s * axis(2);
    ans(1, 1) = (1.0f - c) * axis(1) * axis(1) + c;
    ans(1, 2) = (1.0f - c) * axis(2) * axis(1) - s * axis(0);

    ans(2, 0) = (1.0f - c) * axis(0) * axis(2) - s * axis(1);
    ans(2, 1) = (1.0f - c) * axis(1) * axis(2) + s * axis(0);
    ans(2, 2) = (1.0f - c) * axis(2) * axis(2) + c;

    ans(3, 3) = 1.0f;

    return ans;
}

static Matrix4x4f lookAt(const Vector3f& position, const Vector3f& center, const Vector3f& up) {
    Vector3f f = (center - position).normalized();
    Vector3f s = f.cross(up).normalized();
    Vector3f u = s.cross(f);
    Matrix4x4f ans = Matrix4x4f::Zero();

    ans(0, 0) = s(0);
    ans(0, 1) = s(1);
    ans(0, 2) = s(2);
    ans(0, 3) = -s.dot(position);

    ans(1, 0) = u(0);
    ans(1, 1) = u(1);
    ans(1, 2) = u(2);
    ans(1, 3) = -u.dot(position);

    ans(2, 0) = -f(0);
    ans(2, 1) = -f(1);
    ans(2, 2) = -f(2);
    ans(2, 3) = f.dot(position);

    ans(3, 3) = 1.0f;

    return ans;
}

static Matrix4x4f perspective(const float fovy, const float aspect, const float zNear, const float zFar) {
    float t = std::tan(fovy * 0.5f);
    Matrix4x4f ans = Matrix4x4f::Zero();

    ans(0, 0) = 1.0f / (aspect * t);
    ans(1, 1) = 1.0f / t;
    ans(2, 2) = -(zNear + zFar) / (zFar - zNear);
    ans(2, 3) = -2.0f * zNear * zFar / (zFar - zNear);
    ans(3, 2) = -1.0f;

    return ans;
}

#endif