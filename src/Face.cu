#include "Face.cuh"

Face::Face(const Vertex* vertex0, const Vertex* vertex1, const Vertex* vertex2, const Material* material) :
    vertices{const_cast<Vertex*>(vertex0), const_cast<Vertex*>(vertex1), const_cast<Vertex*>(vertex2)} {
    initialize(material);
}

Face::~Face() {}

void Face::initialize(const Material* material) {   
    if (material != nullptr) {
        Vector2f d1 = vertices[1]->u - vertices[0]->u;
        Vector2f d2 = vertices[2]->u - vertices[0]->u;
        inverse = Matrix2x2f(d1, d2).inverse();
        area = 0.5f * abs(d1.cross(d2));
        mass = area * material->getDensity();
    }
}

void Face::setEdge(const Edge* edge) {
    Node* node0 = edge->nodes[0];
    Node* node1 = edge->nodes[1];
    for (int i = 0; i < 3; i++) {
        Node* n0 = vertices[i]->node;
        Node* n1 = vertices[(i + 1) % 3]->node;
        if (n0 == node0 && n1 == node1 || n1 == node0 && n0 == node1) {
            edges[i] = const_cast<Edge*>(edge);
            return;
        }
    }
}

void Face::setEdges(const Edge* edge0, const Edge* edge1, const Edge* edge2) {
    edges[0] = const_cast<Edge*>(edge0);
    edges[1] = const_cast<Edge*>(edge1);
    edges[2] = const_cast<Edge*>(edge2);
}

void Face::replaceVertex(const Vertex* v, const Vertex* vertex) {
    for (int i = 0; i < 3; i++)
        if (vertices[i] == v) {
            vertices[i] = const_cast<Vertex*>(vertex);
            return;
        }
}

void Face::replaceEdge(const Edge* e, const Edge* edge) {
    for (int i = 0; i < 3; i++)
        if (edges[i] == e) {
            edges[i] = const_cast<Edge*>(edge);
            return;
        }
}

bool Face::isFree() const {
    return vertices[0]->node->isFree && vertices[1]->node->isFree && vertices[2]->node->isFree;
}

bool Face::contain(const Vertex* vertex) const {
    return vertices[0] == vertex || vertices[1] == vertex || vertices[2] == vertex;
}

bool Face::contain(const Edge* edge) const {
    return edges[0] == edge || edges[1] == edge || edges[2] == edge;
}

bool Face::adjacent(const Face* face) const {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (vertices[i]->node == face->vertices[j]->node)
                return true;
    return false;
}

Edge* Face::findEdge(const Vertex* vertex0, const Vertex* vertex1) const {
    for (int i = 0; i < 3; i++) {
        Vertex* v0 = vertices[i];
        Vertex* v1 = vertices[(i + 1) % 3];
        if (v0 == vertex0 && v1 == vertex1 || v1 == vertex0 && v0 == vertex1)
            return edges[i];
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
        const Node* node = vertices[i]->node;
        ans += node->x;
        if (ccd)
            ans += node->x0;
    }
    return ans;
}

Vector3f Face::position(const Vector3f& b) const {
    return b(0) * vertices[0]->node->x + b(1) * vertices[1]->node->x + b(2) * vertices[2]->node->x;
}

Matrix3x2f Face::derivative(const Vector3f& v0, const Vector3f& v1, const Vector3f& v2) const {
    return Matrix3x2f(v1 - v0, v2 - v0) * inverse;
}

Matrix2x2f Face::curvature() const {
    Matrix2x2f ans;
    for (int i = 0; i < 3; i++) {
        Vector2f e = vertices[(i + 1) % 3]->u - vertices[i]->u;
        Vector2f t = Vector2f(-e(1), e(0)).normalized();
        float angle = edges[i]->angle();
        ans -= 0.5f * angle * e.norm() * t.outer(t);
    }
    return ans / area;
}

void Face::update() {
    Vector3f d1 = vertices[1]->node->x - vertices[0]->node->x;
    Vector3f d2 = vertices[2]->node->x - vertices[0]->node->x;
    n = d1.cross(d2).normalized();
    if (!isFree()) {
        area = 0.5f * d1.cross(d2).norm();
        mass = 0.0f;
    }
}
