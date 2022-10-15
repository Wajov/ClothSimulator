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

bool BVHNode::checkImpact(ImpactType type, const Vertex* vertex0, const Vertex* vertex1, const Vertex* vertex2, const Vertex* vertex3, Impact& impact) const {
    // TODO
}

bool BVHNode::checkVertexFaceImpact(const Vertex* vertex, const Face* face, float thickness, Impact& impact) const {
    Vertex* vertex0 = face->getVertex(0);
    Vertex* vertex1 = face->getVertex(1);
    Vertex* vertex2 = face->getVertex(2);
    if (vertex == vertex0 || vertex == vertex1 || vertex == vertex2)
        return false;
    if (!vertexBounds(vertex, true).overlap(faceBounds(face, true), thickness))
        return false;
    
    return checkImpact(VertexFace, vertex, vertex0, vertex1, vertex2, impact);
}

bool BVHNode::checkEdgeEdgeImpact(const Edge* edge0, const Edge* edge1, float thickness, Impact& impact) const {
    Vertex* vertex0 = edge0->getVertex(0);
    Vertex* vertex1 = edge0->getVertex(1);
    Vertex* vertex2 = edge1->gerVertex(0);
    Vertex* vertex3 = edge1->getVertex(1);
    if (vertex0 == vertex2 || vertex0 == vertex3 || vertex1 == vertex2 || vertex1 == vertex3)
        return false;
    if (!edgeBounds(edge0, true).overlap(edgeBounds(edge1, true), thickness))
        return false;
    
    return checkImpact(EdgeEdge, vertex0, vertex1, vertex2, vertex3, impact);
}

void BVHNode::checkImpacts(const Face* face0, const Face* face1, float thickness, std::vector<Impact>& impacts) const {
    Impact impact;
    for (int i = 0; i < 3; i++)
        if (checkVertexFaceCollision(face0->getVertex(i), face1, thickness, impact))
            impacts.push_back(impact);
    for (int i = 0; i < 3; i++)
        if (checkVertexFaceCollision(face1->getVertex(i), face0, thickness, impact))
            impacts.push_back(impact);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (checkEdgeEdgeCollision(face0->getEdge(i), face1->getEdge(j), thickness, impact))
                impacts.push_back(impact);
}

inline bool BVHNode::isLeaf() const {
    return left == nullptr && right == nullptr;
}

void BVHNode::getImpacts(const BVHNode* bvhNode, float thickness, std::vector<Impact>& impacts) const {
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
        checkImpacts(face, bvhNode->face, thickness, impacts);
    else if (isLeaf()) {
        getImpacts(bvhNode->left, thickness, impacts);
        getImpacts(bvhNode->right, thickness, impacts);
    } else {
        left->getImpacts(bvhNode, thickness, impacts);
        right->getImpacts(bvhNode, thickness, impacts);
    }
}
