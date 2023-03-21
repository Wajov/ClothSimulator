#include "Quaternion.cuh"

Quaternion::Quaternion() :
    s(0.0f),
    v() {}

Quaternion::Quaternion(const Vector3f& axis, float angle) {
    if (angle == 0.0f) {
        s = 1.0f;
        v = Vector3f();
    } else {
        s = cos(0.5f * angle);
        v = sin(0.5f * angle) * axis.normalized();
    }
}

Quaternion::Quaternion(float s, float x, float y, float z) :
    s(s),
    v(x, y, z) {}

Quaternion::~Quaternion() {}

Vector3f Quaternion::rotate(const Vector3f& x) const {
    return x * (sqr(s) - v.norm2()) + 2.0f * v * v.dot(x) + 2.0f * v.cross(x) * s;
}

Quaternion Quaternion::operator+(const Quaternion& q) const {
    Quaternion ans;
    ans.s = s + q.s;
    ans.v = v + q.v;
    return ans;
}

Quaternion Quaternion::operator-(const Quaternion& q) const {
    Quaternion ans;
    ans.s = s - q.s;
    ans.v = v - q.v;
    return ans;
}

Quaternion operator*(float s, const Quaternion& q) {
    Quaternion ans;
    ans.s = s * q.s;
    ans.v = s * q.v;
    return ans;
}

Quaternion Quaternion::operator*(float s) const {
    Quaternion ans;
    ans.s = this->s * s;
    ans.v = v * s;
    return ans;
}


Quaternion Quaternion::operator*(const Quaternion& q) const {
    Quaternion ans;
    ans.s = s * q.s - v.dot(q.v);
    ans.v = s * q.v + q.s * v + v.cross(q.v);
    return ans;
}

Quaternion Quaternion::operator/(float s) const {
    Quaternion ans;
    ans.s = this->s / s;
    ans.v = v / s;
    return ans;
}