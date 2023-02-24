#include "BVHNode.cuh"

BVHNode::BVHNode() :
    face(nullptr),
    bounds(),
    parent(nullptr),
    left(nullptr),
    right(nullptr),
    count(0),
    active(true) {}

BVHNode::~BVHNode() {}

float BVHNode::unsignedVertexEdgeDistance(const Vector3f& x, const Vector3f& y0, const Vector3f& y1, Vector3f& n, float& wx, float& wy0, float& wy1) const {
    float t = clamp((x - y0).dot(y1 - y0)/(y1 - y0).dot(y1 - y0), 0.0f, 1.0f);
    Vector3f y = y0 + t * (y1 - y0);
    float d = (x - y).norm();
    n = (x - y).normalized();
    wx = 1.0f;
    wy0 = 1.0f - t;
    wy1 = t;
    return d;
}

float BVHNode::unsignedVertexFaceDistance(const Vector3f& x, const Vector3f& y0, const Vector3f& y1, const Vector3f& y2, Vector3f& n, float* w) const {
    Vector3f nt = (y1 - y0).cross(y2 - y0).normalized();
    float d = abs((x - y0).dot(nt));
    float b0 = mixed(y1 - x, y2 - x, nt);
    float b1 = mixed(y2 - x, y0 - x, nt);
    float b2 = mixed(y0 - x, y1 - x, nt);
    if (b0 >= 0.0f && b1 >= 0.0f && b2 >= 0.0f) {
        n = nt;
        w[0] = 1.0f;
        w[1] = -b0 / (b0 + b1 + b2);
        w[2] = -b1 / (b0 + b1 + b2);
        w[3] = -b2 / (b0 + b1 + b2);
        return d;
    }
    d = INFINITY;
    if (b0 < 0.0f) {
        float dt = unsignedVertexEdgeDistance(x, y1, y2, n, w[0], w[2], w[3]);
        if (dt < d) {
            d = dt;
            w[1] = 0.0f;
        }
    }
    if (b1 < 0.0f) {
        float dt = unsignedVertexEdgeDistance(x, y2, y0, n, w[0], w[3], w[1]);
        if (dt < d) {
            d = dt;
            w[2] = 0.0f;
        }
    }
    if (b2 < 0.0f) {
        float dt = unsignedVertexEdgeDistance(x, y0, y1, n, w[0], w[1], w[2]);
        if (dt < d) {
            d = dt;
            w[3] = 0.0f;
        }
    }
    return d;
}

void BVHNode::checkNearestPoint(const Vector3f& x, const Face* face, NearPoint& point) const {
    Vector3f n;
    float w[4];
    Vector3f x1 = face->vertices[0]->node->x;
    Vector3f x2 = face->vertices[1]->node->x;
    Vector3f x3 = face->vertices[2]->node->x;
    float d = unsignedVertexFaceDistance(x, x1, x2, x3, n, w);

    if (d < point.d) {
        point.d = d;
        point.x = -(w[1] * x1 + w[2] * x2 + w[3] * x3);
    }
}

bool BVHNode::isLeaf() const {
    return left == nullptr && right == nullptr;
}

void BVHNode::setActiveUp(bool active) {
    this->active = active;
    if (parent != nullptr)
        parent->setActiveUp(active);
}

void BVHNode::setActiveDown(bool active) {
    this->active = active;
    if (!isLeaf()) { 
        left->setActiveDown(active);
        right->setActiveDown(active);
    }
}

void BVHNode::findNearestPoint(const Vector3f& x, NearPoint& point) const {
    if (bounds.distance(x) >= point.d)
        return;

    if (isLeaf())
        checkNearestPoint(x, face, point);
    else {
        left->findNearestPoint(x, point);
        right->findNearestPoint(x, point);
    }
}

void BVHNode::update(bool ccd) {
    if (isLeaf())
        bounds = face->bounds(ccd);
    else {
        left->update(ccd);
        right->update(ccd);
        bounds = left->bounds + right->bounds;
    }
}
