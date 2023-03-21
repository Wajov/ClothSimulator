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
        thrust::device_vector<Face*>& faces = const_cast<Mesh*>(mesh)->getFacesGpu();
        int nFaces = faces.size();
        Face** facesPointer = pointer(faces);

        thrust::device_vector<Bounds> bounds(nFaces);
        Bounds* boundsPointer = pointer(bounds);
        computeLeafBounds<<<GRID_SIZE, BLOCK_SIZE>>>(nFaces, facesPointer, ccd, boundsPointer);
        CUDA_CHECK_LAST();
        Bounds objectBounds = thrust::reduce(bounds.begin(), bounds.end(), Bounds());
        Vector3f p = objectBounds.pMin, d = objectBounds.pMax - objectBounds.pMin;

        leaves.resize(nFaces);
        BVHNode* leavesPointer = pointer(leaves);
        thrust::device_vector<unsigned long long> mortonCodes(nFaces);
        unsigned long long* mortonCodesPointer = pointer(mortonCodes);
        initializeLeafNodes<<<GRID_SIZE, BLOCK_SIZE>>>(nFaces, facesPointer, boundsPointer, p, d, leavesPointer, mortonCodesPointer);
        CUDA_CHECK_LAST();
        thrust::sort_by_key(mortonCodes.begin(), mortonCodes.end(), leaves.begin());

        internals.resize(nFaces - 1);
        BVHNode* internalsPointer = pointer(internals);
        initializeInternalNodes<<<GRID_SIZE, BLOCK_SIZE>>>(nFaces - 1, mortonCodesPointer, leavesPointer, internalsPointer);
        CUDA_CHECK_LAST();

        resetCount<<<GRID_SIZE, BLOCK_SIZE>>>(nFaces - 1, internalsPointer);
        CUDA_CHECK_LAST();

        computeInternalBounds<<<GRID_SIZE, BLOCK_SIZE>>>(nFaces, leavesPointer);
        CUDA_CHECK_LAST();

        root = internalsPointer;
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

void BVH::traverse(float thickness, std::function<void(const Face*, const Face*, float)> callback) const {
    root->traverse(thickness, callback);
}

thrust::device_vector<PairFF> BVH::traverse(float thickness) const {
    int nLeaves = leaves.size();
    const BVHNode* leavsPointer = pointer(leaves);
    thrust::device_vector<int> num(nLeaves);
    int* numPointer = pointer(num);
    countPairsSelf<<<GRID_SIZE, BLOCK_SIZE>>>(nLeaves, leavsPointer, root, thickness, numPointer);
    CUDA_CHECK_LAST();

    thrust::inclusive_scan(num.begin(), num.end(), num.begin());
    thrust::device_vector<PairFF> ans(num.back());
    findPairsSelf<<<GRID_SIZE, BLOCK_SIZE>>>(nLeaves, leavsPointer, root, thickness, numPointer, pointer(ans));
    CUDA_CHECK_LAST();

    return ans;
}

void BVH::traverse(const BVH* bvh, float thickness, std::function<void(const Face*, const Face*, float)> callback) const {
    root->traverse(bvh->root, thickness, callback);
}

thrust::device_vector<PairFF> BVH::traverse(const BVH* bvh, float thickness) const {
    int nLeaves = leaves.size();
    const BVHNode* leavesPointer = pointer(leaves);
    thrust::device_vector<int> num(nLeaves);
    int* numPointer = pointer(num);
    countPairs<<<GRID_SIZE, BLOCK_SIZE>>>(nLeaves, leavesPointer, bvh->root, thickness, numPointer);
    CUDA_CHECK_LAST();
    
    thrust::inclusive_scan(num.begin(), num.end(), num.begin());
    thrust::device_vector<PairFF> ans(num.back());
    findPairs<<<GRID_SIZE, BLOCK_SIZE>>>(nLeaves, leavesPointer, bvh->root, thickness, numPointer, pointer(ans));
    CUDA_CHECK_LAST();

    return ans;
}

void BVH::findNearestPoint(const Vector3f& x, NearPoint& point) const {
    root->findNearestPoint(x, point);
}

void BVH::findNearestPoint(const thrust::device_vector<Vector3f>& x, thrust::device_vector<NearPoint>& points) const {
    findNearestPointGpu<<<GRID_SIZE, BLOCK_SIZE>>>(x.size(), pointer(x), root, pointer(points));
    CUDA_CHECK_LAST();
}

void BVH::update() {
    if (!gpu)
        root->update(ccd);
    else {
        resetCount<<<GRID_SIZE, BLOCK_SIZE>>>(internals.size(), pointer(internals));
        CUDA_CHECK_LAST();

        updateGpu<<<GRID_SIZE, BLOCK_SIZE>>>(leaves.size(), pointer(leaves), ccd);
        CUDA_CHECK_LAST();
    }
}
