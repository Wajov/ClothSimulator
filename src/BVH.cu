#include "BVH.cuh"

BVH::BVH(const Mesh* mesh, bool ccd) :
    ccd(ccd) {
    if (!gpu) {
        std::vector<Face*> faces = const_cast<Mesh*>(mesh)->getFaces();
        int nFaces = faces.size();
        std::vector<Bounds> bounds(nFaces);
        std::vector<Vector3f> centers(nFaces);
        for (int i = 0; i < nFaces; i++) {
            bounds[i] = faces[i]->bounds(ccd);
            centers[i] = bounds[i].center();
        }
        index = 0;
        nodes.resize(2 * nFaces - 1);
        initialize(nullptr, 0, nFaces - 1, faces, bounds, centers);
        root = &nodes[0];
    } else {
        thrust::device_vector<Face*> faces = const_cast<Mesh*>(mesh)->getFacesGpu();
        int nFaces = faces.size();
        Face** facesPointer = pointer(faces);

        thrust::device_vector<Bounds> bounds(nFaces);
        Bounds* boundsPointer = pointer(bounds);
        computeLeafBounds<<<GRID_SIZE, BLOCK_SIZE>>>(nFaces, facesPointer, ccd, boundsPointer);
        CUDA_CHECK_LAST();
        Bounds objectBounds = thrust::reduce(bounds.begin(), bounds.end(), Bounds());
        Vector3f p = objectBounds.pMin, d = objectBounds.pMax - objectBounds.pMin;

        leaves.resize(nFaces);
        thrust::device_vector<unsigned long long> mortonCodes(nFaces);
        initializeLeafNodes<<<GRID_SIZE, BLOCK_SIZE>>>(nFaces, facesPointer, boundsPointer, p, d, pointer(leaves), pointer(mortonCodes));
        CUDA_CHECK_LAST();
        thrust::sort_by_key(mortonCodes.begin(), mortonCodes.end(), leaves.begin());

        internals.resize(nFaces - 1);
        initializeInternalNodes<<<GRID_SIZE, BLOCK_SIZE>>>(nFaces - 1, pointer(mortonCodes), pointer(leaves), pointer(internals));
        CUDA_CHECK_LAST();

        computeInternalBounds<<<GRID_SIZE, BLOCK_SIZE>>>(nFaces, pointer(leaves));
        CUDA_CHECK_LAST();

        root = pointer(internals);
    }
}

BVH::~BVH() {}

void BVH::initialize(const BVHNode* parent, int l, int r, std::vector<Face*>& faces, std::vector<Bounds>& bounds, std::vector<Vector3f>& centers) {
    BVHNode& node = nodes[index++];
    node.parent = const_cast<BVHNode*>(parent);
    if (l == r) {
        node.face = faces[l];
        node.bounds = bounds[l];
        node.left = nullptr;
        node.right = nullptr;
        for (int i = 0; i < 3; i++)
            adjacents[node.face->vertices[i]->node].push_back(&node);
    } else {
        for (int i = l; i <= r; i++)
            node.bounds += bounds[i];
        
        if (r - l == 1) {
            node.left = &nodes[index];
            initialize(&node, l, l, faces, bounds, centers);
            node.right = &nodes[index];
            initialize(&node, r, r, faces, bounds, centers);
        } else {
            Vector3f center = node.bounds.center();
            int axis = node.bounds.majorAxis();
            int lt = l, rt = r;
            for (int i = l; i <= r; i++)
                if (centers[i](axis) > center(axis))
                    lt++;
                else {
                    mySwap(faces[lt], faces[rt]);
                    mySwap(bounds[lt], bounds[rt]);
                    mySwap(centers[lt], centers[rt]);
                    rt--;
                }

            if (lt > l && lt < r) {
                node.left = &nodes[index];
                initialize(&node, l, rt, faces, bounds, centers);
                node.right = &nodes[index];
                initialize(&node, lt, r, faces, bounds, centers);
            } else {
                int mid = l + r >> 1;
                node.left = &nodes[index];
                initialize(&node, l, mid, faces, bounds, centers);
                node.right = &nodes[index];
                initialize(&node, mid + 1, r, faces, bounds, centers);
            }
        }
    }
}

BVHNode* BVH::getRoot() const {
    return root;
}

bool BVH::contain(const Node* node) const {
    return adjacents.find(const_cast<Node*>(node)) != adjacents.end();
}

void BVH::setAllActive(bool active) {
    root->setActiveDown(active);
}

void BVH::setActive(const Node* node, bool active) {
    std::vector<BVHNode*>& adjacentNodes = this->adjacents[const_cast<Node*>(node)];
    for (BVHNode* node : adjacentNodes)
        node->setActiveUp(active);
}

void BVH::findNearestPoint(const Vector3f& x, NearPoint& point) const {
    root->findNearestPoint(x, point);
}

void BVH::update() {
    root->update(ccd);
}
