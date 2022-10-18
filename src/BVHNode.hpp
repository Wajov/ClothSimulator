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
    float signedVertexFaceDistance(const Vector3f& x, const Vector3f& y0, const Vector3f& y1, const Vector3f& y2, Vector3f& n, float* w) const;
    float signedEdgeEdgeDistance(const Vector3f& x0, const Vector3f& x1, const Vector3f& y0, const Vector3f& y1, Vector3f& n, float* w) const;
    bool checkImpact(ImpactType type, const Vertex* vertex0, const Vertex* vertex1, const Vertex* vertex2, const Vertex* vertex3, Impact& impact) const;
    bool checkVertexFaceImpact(const Vertex* vertex, const Face* face, float thickness, Impact& impact) const;
    bool checkEdgeEdgeImpact(const Edge* edge0, const Edge* edge1, float thickness, Impact& impact) const;
    void checkImpacts(const Face* face0, const Face* face1, float thickness, std::vector<Impact>& impacts) const;

public:
    BVHNode(int l, int r, std::vector<Face*>& faces, std::vector<Bounds>& bounds, std::vector<Vector3f>& centers);
    ~BVHNode();
    inline bool isLeaf() const;
    void findImpacts(float thickness, std::vector<Impact>& impacts) const;
    void findImpacts(const BVHNode* bvhNode, float thickness, std::vector<Impact>& impacts) const;
};

#endif