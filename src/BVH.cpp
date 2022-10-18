#include "BVH.hpp"

BVH::BVH(const Mesh* mesh, bool ccd) :
    ccd(ccd) {
    std::vector<Face*> faces = const_cast<Mesh*>(mesh)->getFaces();
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
    root = new BVHNode(nullptr, 0, n - 1, faces, bounds, centers, leaves);

    std::vector<Vertex>& vertices = const_cast<Mesh*>(mesh)->getVertices();
    for (Vertex& vertex : vertices)
        this->vertices.insert(&vertex);

    for (Face* face : faces)
        for (int i = 0; i < 3; i++)
            adjacents[face->getVertex(i)].push_back(face);
}

BVH::~BVH() {
    delete root;
}

bool BVH::contain(const Vertex* vertex) const {
    return vertices.find(const_cast<Vertex*>(vertex)) != vertices.end();
}

void BVH::setAllActive(bool active) {
    root->setActiveDown(active);
}

void BVH::setActive(const Vertex* vertex, bool active) {
    std::vector<Face*>& adjacents = this->adjacents[const_cast<Vertex*>(vertex)];
    for (Face* face : adjacents)
        leaves[face]->setActiveUp(active);
}

void BVH::findImpacts(float thickness, std::vector<Impact>& impacts) const {
    root->findImpacts(thickness, impacts);
}

void BVH::findImpacts(const BVH* bvh, float thickness, std::vector<Impact>& impacts) const {
    root->findImpacts(bvh->root, thickness, impacts);
}

void BVH::update() {
    root->update(ccd);
}
