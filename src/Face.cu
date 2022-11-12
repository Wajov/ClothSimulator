#include "Face.cuh"

Face::Face(const Vertex* vertex0, const Vertex* vertex1, const Vertex* vertex2, const Material* material) :
    vertices{const_cast<Vertex*>(vertex0), const_cast<Vertex*>(vertex1), const_cast<Vertex*>(vertex2)} {
    initialize(material);
}

Face::~Face() {}

void Face::initialize(const Material* material) {
    Vector2f d1 = vertices[1]->u - vertices[0]->u;
    Vector2f d2 = vertices[2]->u - vertices[0]->u;
    inverse = Matrix2x2f(d1, d2).inverse();
    area = 0.5f * std::abs(d1.cross(d2));
    if (material != nullptr)
        mass = material->getDensity() * material->getThicken() * area;
}

Vertex* Face::getVertex(int index) const {
    return vertices[index];
}

void Face::replaceVertex(const Vertex* v, const Vertex* vertex) {
    for (int i = 0; i < 3; i++)
        if (vertices[i] == v) {
            vertices[i] = const_cast<Vertex*>(vertex);
            return;
        }
}

Edge* Face::getEdge(int index) const {
    return edges[index];
}

void Face::setEdge(const Edge* edge) {
    for (int i = 0; i < 2; i++)
        if (vertices[0] == edge->getVertex(0) && vertices[1] == edge->getVertex(1) || vertices[1] == edge->getVertex(0) && vertices[0] == edge->getVertex(1)) {
            edges[i] = const_cast<Edge*>(edge);
            return;
        }
    edges[2] = const_cast<Edge*>(edge);
}

void Face::setEdges(const Edge* edge0, const Edge* edge1, const Edge* edge2) {
    edges[0] = const_cast<Edge*>(edge0);
    edges[1] = const_cast<Edge*>(edge1);
    edges[2] = const_cast<Edge*>(edge2);
}

void Face::replaceEdge(const Edge* e, const Edge* edge) {
    for (int i = 0; i < 3; i++)
        if (edges[i] == e) {
            edges[i] = const_cast<Edge*>(edge);
            return;
        }
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

int Face::sequence(const Edge* edge) const {
    Vertex* vertex0 = edge->getVertex(0);
    Vertex* vertex1 = edge->getVertex(1);
    for (int i = 0; i < 2; i++)
        if (vertices[i] == vertex0 && vertices[i + 1] == vertex1)
            return 0;
    return vertices[2] == vertex0 && vertices[0] == vertex1 ? 0 : 1;
}

bool Face::contain(const Edge* edge) const {
    for (int i = 0; i < 3; i++)
        if (edges[i] == edge)
            return true;
    return false;
}

Edge* Face::findEdge(const Vertex* vertex0, const Vertex* vertex1) const {
    for (int i = 0; i < 3; i++) {
        Edge* edge = edges[i];
        Vertex* v0 = edge->getVertex(0);
        Vertex* v1 = edge->getVertex(1);
        if (vertex0 == v0 && vertex1 == v1 || vertex0 == v1 && vertex1 == v0)
            return edge;
    }
    return nullptr;
}

Edge* Face::findOpposite(const Vertex* vertex) const {
    for (int i = 0; i < 3; i++)
        if (vertices[i] == vertex)
            return edges[(i + 1) % 3];
    return nullptr;
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
    return Matrix3x2f(v1 - v0, v2 - v0) * inverse;
}

Matrix2x2f Face::curvature() const {
    Matrix2x2f ans;
    for (int i = 0; i < 3; i++) {
        Edge* edge = edges[i];
        Vector2f e = edge->getVertex(1)->u - edge->getVertex(0)->u;
        Vector2f t = Vector2f(-e(1), e(0)).normalized();
        float angle = edge->getAngle();
        ans -= 0.5f * angle * e.norm() * t.outer(t);
    }
    return ans / area;
}

void Face::update() {
    Vector3f d1 = vertices[1]->x - vertices[0]->x;
    Vector3f d2 = vertices[2]->x - vertices[0]->x;
    normal = d1.cross(d2).normalized();
}
