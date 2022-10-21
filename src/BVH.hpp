#ifndef BVH_HPP
#define BVH_HPP

#include <vector>
#include <unordered_map>

#include "BVHNode.hpp"
#include "Vertex.hpp"
#include "Face.hpp"
#include "Mesh.hpp"
#include "Impact.hpp"
#include "NearPoint.hpp"

class BVH {
private:
    bool ccd;
    BVHNode* root;
    std::vector<Vertex*> vertices;
    std::vector<std::vector<Face*>> adjacents;
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