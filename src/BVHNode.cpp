#include "BVHNode.hpp"

BVHNode::BVHNode(int l, int r, std::vector<Face*>& faces, std::vector<Bounds>& bounds, std::vector<Vector3f>& centers) {
    face = nullptr;
    active = true;
    if (l == r) {
        face = faces[l];
        this->bounds = bounds[l];
        left = nullptr;
        right = nullptr;
    } else {
        for (int i = l; i <= r; i++)
            this->bounds += bounds[i];
        
        if (r - l == 1) {
            left = new BVHNode(l, l, faces, bounds, centers);
            right = new BVHNode(r, r, faces, bounds, centers);
        } else {
            Vector3f center = this->bounds.center();
            int index = this->bounds.longestIndex();
            int lt = l, rt = r;
            for (int i = l; i <= r; i++)
                if (centers[i](index) > center(index))
                    lt++;
                else {
                    std::swap(faces[lt], faces[rt]);
                    std::swap(bounds[lt], bounds[rt]);
                    std::swap(centers[lt], centers[rt]);
                    rt--;
                }

            if (lt > l && lt < r) {
                left = new BVHNode(l, lt, faces, bounds, centers);
                right = new BVHNode(rt, r, faces, bounds, centers);
            } else {
                int mid = l + r >> 1;
                left = new BVHNode(l, mid, faces, bounds, centers);
                right = new BVHNode(mid + 1, r, faces, bounds, centers);
            }
        }
    }
}

BVHNode::~BVHNode() {
    delete left;
    delete right;
}

void BVHNode::checkImpacts(const Face* face0, const Face* face1, std::vector<Impact>& impacts) const {
    Impact impact;
    // for (int i = 0; i < 3; i++)
    //     if (checkVertexFaceCollision(face0->getVertex(i), face1, impact))
    //         impacts.push_back(impact);
    // for (int i = 0; i < 3; i++)
    //     if (checkVertexFaceCollision(face1->getVertex(i), face0, impact))
    //         impacts.push_back(impact);
    // for (int i = 0; i < 3; i++)
    //     for (int j = 0; j < 3; j++)
    //         if (checkEdgeEdgeCollision(face0->getEdge(i), face1->getEdge(j), impact))
    //             impacts.push_back(impact);
}

inline bool BVHNode::isLeaf() const {
    return left == nullptr && right == nullptr;
}

void BVHNode::getImpacts(float thickness, std::vector<Impact>& impacts) const {
    if (isLeaf() || !active)
        return;

    left->getImpacts(thickness, impacts);
    right->getImpacts(thickness, impacts);
    left->getImpacts(right, thickness, impacts);
}

void BVHNode::getImpacts(const BVHNode* bvhNode, float thickness, std::vector<Impact>& impacts) const {
    if (!active && !bvhNode->active)
        return;
    if (!bounds.overlap(bvhNode->bounds, thickness))
        return;

    if (isLeaf() && bvhNode->isLeaf())
        checkImpacts(face, bvhNode->face, impacts);
    else if (isLeaf()) {
        getImpacts(bvhNode->left, thickness, impacts);
        getImpacts(bvhNode->right, thickness, impacts);
    } else {
        left->getImpacts(bvhNode, thickness, impacts);
        right->getImpacts(bvhNode, thickness, impacts);
    }
}
