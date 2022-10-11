#ifndef BVH_HPP
#define BVH_HPP

#include "BVHNode.hpp"
#include "Face.hpp"
#include "Mesh.hpp"

class BVH {
private:
    BVHNode* root;

public:
    BVH(const Mesh* mesh, bool ccd);
    ~BVH();
};

#endif