#ifndef TRANSFORM_HPP
#define TRANSFORM_HPP

#include "json/json.h"

#include "TypeHelper.hpp"
#include "JsonHelper.hpp"

class Transform {
private:
    Matrix4x4f matrix;

public:
    Transform(const Json::Value& json);
    ~Transform();
    Vector3f applyTo(const Vector3f& v) const;
    static Matrix4x4f scale(float scaling);
    static Matrix4x4f rotate(const Vector3f& v, float angle);
    static Matrix4x4f translate(const Vector3f& v);
    static Matrix4x4f lookAt(const Vector3f& position, const Vector3f& center, const Vector3f& up);
    static Matrix4x4f perspective(float fovy, float aspect, float zNear, float zFar);
    Matrix4x4f getMatrix() const;
};

#endif