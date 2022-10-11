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