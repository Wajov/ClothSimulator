#ifndef BVH_HPP
#define BVH_HPP

#include "BVHNode.hpp"
#include "Face.hpp"
#include "Mesh.hpp"
#include "Impact.hpp"

class BVH {
private:
    BVHNode* root;

public:
    BVH(const Mesh* mesh, bool ccd);
    ~BVH();
    void getImpacts(float thickness, std::vector<Impact>& impacts) const;
    void getImpacts(const BVH* bvh, float thickness, std::vector<Impact>& impacts) const;
};

#endif