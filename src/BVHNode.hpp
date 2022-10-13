#ifndef BVH_NODE_HPP
#define BVH_NODE_HPP

#include "Bounds.hpp"
#include "Face.hpp"
#include "Impact.hpp"

class BVHNode {
private:
    Face* face;
    Bounds bounds;
    BVHNode* left, * right;
    bool active;
    void checkImpacts(const Face* face0, const Face* face1, std::vector<Impact>& impacts) const;

public:
    BVHNode(int l, int r, std::vector<Face*>& faces, std::vector<Bounds>& bounds, std::vector<Vector3f>& centers);
    ~BVHNode();
    inline bool isLeaf() const;
    void getImpacts(float thickness, std::vector<Impact>& impacts) const;
    void getImpacts(const BVHNode* bvhNode, float thickness, std::vector<Impact>& impacts) const;
};

#endif