#include "Face.hpp"

Face::Face(const Vertex* vertex0, const Vertex* vertex1, const Vertex* vertex2) :
    vertices{const_cast<Vertex*>(vertex0), const_cast<Vertex*>(vertex1), const_cast<Vertex*>(vertex2)} {
    Vector3f d1 = vertex1->u - vertex0->u;
    Vector3f d2 = vertex2->u - vertex0->u;
    Vector3f n = d1.cross(d2).normalized();
    inverse = concatenateToMatrix(d1, d2, n).inverse();
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

Matrix3x3f Face::getInverse() const {
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

void Face::updateData(const Material* material) {
    Vector3f d1, d2;

    d1 = vertices[1]->x - vertices[0]->x;
    d2 = vertices[2]->x - vertices[0]->x;
    normal = d1.cross(d2).normalized();

    d1 = vertices[1]->u - vertices[0]->u;
    d2 = vertices[2]->u - vertices[0]->u;
    area = 0.5f * d1.cross(d2).norm();
    mass = material->getDensity() * material->getThicken() * area;
}
