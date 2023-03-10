#include "BackupFace.cuh"

BackupFace::BackupFace() {}

BackupFace::~BackupFace() {}

Vector3f BackupFace::barycentricCoordinates(const Vector2f& u) const {
    Vector2f x = Matrix2x2f(this->u[0] - this->u[2], this->u[1] - this->u[2]).inverse() * (u - this->u[2]);
    return Vector3f(x(0), x(1), 1.0f - x(0) - x(1));
}

Vector3f BackupFace::position(const Vector3f& b) const {
    return b(0) * x[0] + b(1) * x[1] + b(2) * x[2];
}