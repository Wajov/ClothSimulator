#include "BVH.hpp"

BVH::BVH(const Mesh* mesh, bool ccd) {
    std::vector<Face*> faces = mesh->getFaces();
    int n = faces.size();
    std::vector<Bounds> bounds(n);
    std::vector<Vector3f> centers(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 3; j++)
            bounds[i] += faces[i]->getVertex(j)->x;
        if (ccd) {
            Bounds ccdBounds;
            for (int j = 0; j < 3; j++)
                ccdBounds += faces[i]->getVertex(j)->x0;
            centers[i] = 0.5f * (bounds[i].center() + ccdBounds.center());
            bounds[i] += ccdBounds;
        } else
            centers[i] = bounds[i].center();
    }

    root = new BVHNode(0, n - 1, faces, bounds, centers);
}

BVH::~BVH() {
    delete root;
}
void BVH::getImpacts(float thickness, std::vector<Impact>& impacts) const {
    root->getImpacts(thickness, impacts);
}

void BVH::getImpacts(const BVH* bvh, float thickness, std::vector<Impact>& impacts) const {
    root->getImpacts(bvh->root, thickness, impacts);
}