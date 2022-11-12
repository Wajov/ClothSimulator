#ifndef BVH_HPP
#define BVH_HPP

#include <vector>
#include <unordered_map>

#include "BVHNode.hpp"
#include "Vertex.cuh"
#include "Face.cuh"
#include "Mesh.cuh"
#include "Impact.hpp"
#include "NearPoint.hpp"

class BVH {
private:
    bool ccd;
    BVHNode* root;
    std::vector<Vertex*> vertices;
    std::unordered_map<Vertex*, std::vector<Face*>> adjacents;
    std::unordered_map<Face*, BVHNode*> leaves;

public:
    BVH(const Mesh* mesh, bool ccd);
    ~BVH();
    bool contain(const Vertex* vertex) const;
    void setAllActive(bool active);
    void setActive(const Vertex* vertex, bool active);
    void findImpacts(float thickness, std::vector<Impact>& impacts) const;
    void findImpacts(const BVH* bvh, float thickness, std::vector<Impact>& impacts) const;
    void findNearestPoint(const Vector3f& x, NearPoint& point) const;
    void update();
};

#endif