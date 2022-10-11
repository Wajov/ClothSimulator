#ifndef BVH_NODE_HPP
#define BVH_NODE_HPP

#include "Bounds.hpp"
#include "Face.hpp"

class BVHNode {
private:
    Face* face;
    Bounds bounds;
    BVHNode* left, * right;
    bool active;

public:
    BVHNode(int l, int r, std::vector<Face*>& faces, std::vector<Bounds>& bounds, std::vector<Vector3f>& centers);
    ~BVHNode();
};

#endif