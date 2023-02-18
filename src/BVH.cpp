#include "BVH.hpp"

BVH::BVH(const Mesh* mesh, bool ccd) :
    ccd(ccd) {
    std::vector<Face*> faces = const_cast<Mesh*>(mesh)->getFaces();
    int nFaces = faces.size();
    std::vector<Bounds> bounds(nFaces);
    std::vector<Vector3f> centers(nFaces);
    for (int i = 0; i < nFaces; i++) {
        for (int j = 0; j < 3; j++)
            bounds[i] += faces[i]->vertices[j]->node->x;
        if (ccd) {
            Bounds ccdBounds;
            for (int j = 0; j < 3; j++)
                ccdBounds += faces[i]->vertices[j]->node->x0;
            centers[i] = 0.5f * (bounds[i].center() + ccdBounds.center());
            bounds[i] += ccdBounds;
        } else
            centers[i] = bounds[i].center();
    }
    root = new BVHNode(nullptr, 0, nFaces - 1, faces, bounds, centers, adjacents);
}

BVH::~BVH() {
    delete root;
}

bool BVH::contain(const Node* node) const {
    return adjacents.find(const_cast<Node*>(node)) != adjacents.end();
}

void BVH::setAllActive(bool active) {
    root->setActiveDown(active);
}

void BVH::setActive(const Node* node, bool active) {
    std::vector<BVHNode*>& bvhNodes = this->adjacents[const_cast<Node*>(node)];
    for (BVHNode* bvhNode : bvhNodes)
        bvhNode->setActiveUp(active);
}

void BVH::traverse(float thickness, std::function<void(const Face*, const Face*, float)> callback) {
    root->traverse(thickness, callback);
}

void BVH::traverse(const BVH* bvh, float thickness, std::function<void(const Face*, const Face*, float)> callback) {
    root->traverse(bvh->root, thickness, callback);
}

void BVH::findNearestPoint(const Vector3f& x, NearPoint& point) const {
    root->findNearestPoint(x, point);
}

void BVH::update() {
    root->update(ccd);
}
