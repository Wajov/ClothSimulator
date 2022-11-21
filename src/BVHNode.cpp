#include "BVHNode.hpp"

BVHNode::BVHNode(BVHNode* parent, int l, int r, std::vector<Face*>& faces, std::vector<Bounds>& bounds, std::vector<Vector3f>& centers, std::unordered_map<Face*, BVHNode*>& leaves) : 
    parent(parent) {
    face = nullptr;
    active = true;
    if (l == r) {
        face = faces[l];
        this->bounds = bounds[l];
        left = nullptr;
        right = nullptr;
        leaves[face] = this;
    } else {
        for (int i = l; i <= r; i++)
            this->bounds += bounds[i];
        
        if (r - l == 1) {
            left = new BVHNode(this, l, l, faces, bounds, centers, leaves);
            right = new BVHNode(this, r, r, faces, bounds, centers, leaves);
        } else {
            Vector3f center = this->bounds.center();
            int index = this->bounds.longestIndex();
            int lt = l, rt = r;
            for (int i = l; i <= r; i++)
                if (centers[i](index) > center(index))
                    lt++;
                else {
                    mySwap(faces[lt], faces[rt]);
                    mySwap(bounds[lt], bounds[rt]);
                    mySwap(centers[lt], centers[rt]);
                    rt--;
                }

            if (lt > l && lt < r) {
                left = new BVHNode(this, l, rt, faces, bounds, centers, leaves);
                right = new BVHNode(this, lt, r, faces, bounds, centers, leaves);
            } else {
                int mid = l + r >> 1;
                left = new BVHNode(this, l, mid, faces, bounds, centers, leaves);
                right = new BVHNode(this, mid + 1, r, faces, bounds, centers, leaves);
            }
        }
    }
}

BVHNode::~BVHNode() {
    delete left;
    delete right;
}

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
    Vector3f x1 = face->getVertex(0)->x;
    Vector3f x2 = face->getVertex(1)->x;
    Vector3f x3 = face->getVertex(2)->x;
    float d = unsignedVertexFaceDistance(x, x1, x2, x3, n, w);

    if (d < point.d) {
        point.d = d;
        point.x = -(w[1] * x1 + w[2] * x2 + w[3] * x3);
    }
}

inline bool BVHNode::isLeaf() const {
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

void BVHNode::traverse(float thickness, std::function<void(const Face*, const Face*, float)> callback) {
    if (isLeaf() || !active)
        return;

    left->traverse(thickness, callback);
    right->traverse(thickness, callback);
    left->traverse(right, thickness, callback);
}

void BVHNode::traverse(const BVHNode* bvhNode, float thickness, std::function<void(const Face*, const Face*, float)> callback) {
    if (!active && !bvhNode->active)
        return;
    if (!bounds.overlap(bvhNode->bounds, thickness))
        return;

    if (isLeaf() && bvhNode->isLeaf())
        callback(face, bvhNode->face, thickness);
    else if (isLeaf()) {
        traverse(bvhNode->left, thickness, callback);
        traverse(bvhNode->right, thickness, callback);
    } else {
        left->traverse(bvhNode, thickness, callback);
        right->traverse(bvhNode, thickness, callback);
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
