#include "BVHNode.cuh"

BVHNode::BVHNode() :
    face(nullptr),
    bounds(),
    parent(nullptr),
    left(nullptr),
    right(nullptr),
    active(true) {}

BVHNode::~BVHNode() {}

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

bool BVHNode::isLeaf() const {
    return left == nullptr && right == nullptr;
}

void BVHNode::traverse(float thickness, std::function<void(const Face*, const Face*, float)> callback) const {
    if (isLeaf() || !active)
        return;

    left->traverse(thickness, callback);
    right->traverse(thickness, callback);
    left->traverse(right, thickness, callback);
}

void BVHNode::traverse(const BVHNode* node, float thickness, std::function<void(const Face*, const Face*, float)> callback) const {
    if (!active && !node->active)
        return;
    if (!bounds.overlap(node->bounds, thickness))
        return;

    if (isLeaf() && node->isLeaf())
        callback(face, node->face, thickness);
    else if (isLeaf()) {
        traverse(node->left, thickness, callback);
        traverse(node->right, thickness, callback);
    } else {
        left->traverse(node, thickness, callback);
        right->traverse(node, thickness, callback);
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
