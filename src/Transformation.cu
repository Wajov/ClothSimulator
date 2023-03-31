#include "Transformation.cuh"

Transformation::Transformation() :
    scaling(1.0f),
    translation(),
    rotation(Vector3f(), 0.0f) {}

Transformation::Transformation(const Json::Value& json) {
    scaling = parseFloat(json["scale"], 1.0f);
    translation = parseVector3f(json["translate"]);
    Vector4f r = parseVector4f(json["rotate"]);
    rotation = Quaternion(Vector3f(r(1), r(2), r(3)), r(0) * M_PI / 180.0f);
}

Transformation::Transformation(const Vector3f& translation, const Quaternion& rotation) :
    scaling(1.0f),
    translation(translation),
    rotation(rotation) {}

Transformation::~Transformation() {}

Transformation Transformation::operator+(const Transformation& t) const {
    Transformation ans;
    ans.scaling = scaling + t.scaling;
    ans.translation = translation + t.translation;
    ans.rotation = rotation + t.rotation;
    return ans;
}

Transformation Transformation::operator-(const Transformation& t) const {
    Transformation ans;
    ans.scaling = scaling - t.scaling;
    ans.translation = translation - t.translation;
    ans.rotation = rotation - t.rotation;
    return ans;
}

Transformation operator*(float s, const Transformation& t) {
    Transformation ans;
    ans.scaling = s * t.scaling;
    ans.translation = s * t.translation;
    ans.rotation = s * t.rotation;
    return ans;
}

Transformation Transformation::operator*(float s) const {
    Transformation ans;
    ans.scaling = scaling * s;
    ans.translation = translation * s;
    ans.rotation = rotation * s;
    return ans;
}

Transformation Transformation::operator*(const Transformation& t) const {
    Transformation ans;
    ans.scaling = scaling * t.scaling;
    ans.translation = translation + rotation.rotate(t.scaling * t.translation);
    ans.rotation = rotation * t.rotation;
    return ans;
}

Transformation Transformation::operator/(float s) const {
    Transformation ans;
    ans.scaling = scaling / s;
    ans.translation = translation / s;
    ans.rotation = rotation / s;
    return ans;
}

__host__ __device__ Transformation Transformation::inverse() const {
    Transformation ans;
    ans.scaling = 1.0f / scaling;
    ans.rotation = rotation.inverse();
    ans.translation = Vector3f() - ans.rotation.rotate(ans.scaling * translation);
    return ans;
}

Vector2f Transformation::applyToUV(const Vector2f& u) const {
    return scaling * u;
}

Vector3f Transformation::applyToPoint(const Vector3f& p) const {
    return scaling * rotation.rotate(p) + translation;
}

Vector3f Transformation::applyToVector(const Vector3f& v) const {
    return rotation.rotate(v);
}