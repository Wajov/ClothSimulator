#include "Face.hpp"

Face::Face(const Vertex* v0, const Vertex* v1, const Vertex* v2) :
    v0(const_cast<Vertex*>(v0)),
    v1(const_cast<Vertex*>(v1)),
    v2(const_cast<Vertex*>(v2)) {
    Vector3f d1 = v1->u - v0->u;
    Vector3f d2 = v2->u - v0->u;
    Vector3f n = d1.cross(d2).normalized();
    inverse = concatenateToMatrix(d1, d2, n).inverse();
}

Face::~Face() {}

Vertex* Face::getV0() const {
    return v0;
}

Vertex* Face::getV1() const {
    return v1;
}

Vertex* Face::getV2() const {
    return v2;
}

Vector3f Face::getNormal() const {
    return normal;
}

Matrix3x3f Face::getInverse() const {
    return inverse;
}

float Face::getArea() const {
    return area;
}

float Face::getMass() const {
    return mass;
}

void Face::updateData(const Material* material) {
    Vector3f d1, d2;

    d1 = v1->x - v0->x;
    d2 = v2->x - v0->x;
    normal = d1.cross(d2).normalized();

    d1 = v1->u - v0->u;
    d2 = v2->u - v0->u;
    area = 0.5f * d1.cross(d2).norm();
    mass = material->getDensity() * material->getThicken() * area;
}