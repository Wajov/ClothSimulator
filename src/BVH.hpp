#ifndef BVH_HPP
#define BVH_HPP

#include <vector>
#include <unordered_set>
#include <unordered_map>

#include "BVHNode.hpp"
#include "Vertex.hpp"
#include "Face.hpp"
#include "Mesh.hpp"
#include "Impact.hpp"

class BVH {
private:
    BVHNode* root;
    std::unordered_set<Vertex*> vertices;
    std::unordered_map<Vertex*, std::vector<Face*>> adjacents;

public:
    BVH(const Mesh* mesh, bool ccd);
    ~BVH();
    bool contain(const Vertex* vertex) const;
    void setAllActive(const Vertex* vertex);
    void setActive(const Vertex* vertex);
    void findImpacts(float thickness, std::vector<Impact>& impacts) const;
    void findImpacts(const BVH* bvh, float thickness, std::vector<Impact>& impacts) const;
};

#endif