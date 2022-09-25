#ifndef FACE_HPP
#define FACE_HPP

#include "TypeHelper.hpp"
#include "MathHelper.hpp"
#include "Vertex.hpp"
#include "Material.hpp"

class Face {
private:
    Vertex* v0, * v1, * v2;
    Vector3f normal;
    Matrix3x3f inverse;
    float area, mass;

public:
    Face(const Vertex* v0, const Vertex* v1, const Vertex* v2);
    ~Face();
    Vertex* getV0() const;
    Vertex* getV1() const;
    Vertex* getV2() const;
    Vector3f getNormal() const;
    Matrix3x3f getInverse() const;
    float getArea() const;
    float getMass() const;
    void updateData(const Material* material);
};

#endif