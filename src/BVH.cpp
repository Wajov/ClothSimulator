#include "BVH.hpp"

BVH::BVH(const Mesh* mesh, bool ccd) {
    std::vector<Face*> faces = mesh->getFaces();
    int n = faces.size();
    std::vector<Bounds> bounds(n);
    std::vector<Vector3f> centers(n);
    for (int i = 0; i < n; i++) {
        bounds[i] += faces[i]->getV0()->x;
        bounds[i] += faces[i]->getV1()->x;
        bounds[i] += faces[i]->getV2()->x;
        if (ccd) {
            Bounds ccdBounds;
            ccdBounds += faces[i]->getV0()->x0;
            ccdBounds += faces[i]->getV1()->x0;
            ccdBounds += faces[i]->getV2()->x0;
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