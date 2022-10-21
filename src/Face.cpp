#include "Face.hpp"

Face::Face(const Vertex* vertex0, const Vertex* vertex1, const Vertex* vertex2, const Material* material) :
    vertices{const_cast<Vertex*>(vertex0), const_cast<Vertex*>(vertex1), const_cast<Vertex*>(vertex2)} {
    Vector2f d1 = vertex1->u - vertex0->u;
    Vector2f d2 = vertex2->u - vertex0->u;
    inverse = concatenateToMatrix(d1, d2).inverse();
    area = 0.5f * std::abs(d1(0) * d2(1) - d1(1) * d2(0));
    if (material != nullptr)
        mass = material->getDensity() * material->getThicken() * area;
}

Face::~Face() {}

Vertex* Face::getVertex(int index) const {
    return vertices[index];
}

Edge* Face::getEdge(int index) const {
    return edges[index];
}

void Face::setEdges(const Edge* edge0, const Edge* edge1, const Edge* edge2) {
    edges = {const_cast<Edge*>(edge0), const_cast<Edge*>(edge1), const_cast<Edge*>(edge2)};
}

Vector3f Face::getNormal() const {
    return normal;
}

Matrix2x2f Face::getInverse() const {
    return inverse;
}

float Face::getArea() const {
    return area;
}

float Face::getMass() const {
    return mass;
}

Bounds Face::bounds(bool ccd) const {
    Bounds ans;
    for (int i = 0; i < 3; i++) {
        const Vertex* vertex = vertices[i];
        ans += vertex->x;
        if (ccd)
            ans += vertex->x0;
    }
    return ans;
}

Matrix3x2f Face::derivative(const Vector3f& v0, const Vector3f& v1, const Vector3f& v2) const {
    return concatenateToMatrix(v1 - v0, v2 - v0) * inverse;
}

Matrix2x2f Face::curvature() const {
    Matrix2x2f ans = Matrix2x2f::Zero();
    for (int i = 0; i < 3; i++) {
        Edge* edge = edges[i];
        Vector2f e = edge->getVertex(1)->u - edge->getVertex(0)->u;
        Vector2f t = Vector2f(-e(1), e(0)).normalized();
        float angle = edge->getAngle();
        ans -= 0.5f * angle * e.norm() * t * t.transpose();
    }
    return ans / area;
}

void Face::update() {
    Vector3f d1 = vertices[1]->x - vertices[0]->x;
    Vector3f d2 = vertices[2]->x - vertices[0]->x;
    normal = d1.cross(d2).normalized();
}
