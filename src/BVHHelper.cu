#include "BVHHelper.cuh"

__global__ void computeLeafBounds(int nFaces, const Face* const* faces, bool ccd, Bounds* bounds) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads)
        bounds[i] = faces[i]->bounds(ccd);
}

__device__ unsigned int expandBits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ unsigned int mortonCode(const Vector3f& v, const Vector3f& p, const Vector3f& d) {
    float x = d(0) == 0.0f ? 0.0f : (v(0) - p(0)) / d(0);
    float y = d(1) == 0.0f ? 0.0f : (v(1) - p(1)) / d(1);
    float z = d(2) == 0.0f ? 0.0f : (v(2) - p(2)) / d(2);
    x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return (xx << 2) + (yy << 1) + zz;
}

__global__ void initializeLeafNodes(int nNodes, const Face* const* faces, const Bounds* bounds, const Vector3f p, const Vector3f d, BVHNode* nodes, unsigned long long* mortonCodes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads) {
        const Bounds& b = bounds[i];
        BVHNode& node = nodes[i];
        node.face = const_cast<Face*>(faces[i]);
        node.bounds = b;
        unsigned long long code = mortonCode(b.center(), p, d);
        mortonCodes[i] = (code << 32) | static_cast<unsigned long long>(i);
    }
}

__device__ int commonUpperBits(unsigned long long a, unsigned long long b) {
    return __clzll(a ^ b);
}

__device__ void findRange(int nNodes, const unsigned long long* mortonCodes, int i, int& left, int& right) {
    if (i == 0) {
        left = 0;
        right = nNodes;
        return;
    }

    unsigned long long code = mortonCodes[i];
    int dl = commonUpperBits(code, mortonCodes[i - 1]);
    int dr = commonUpperBits(code, mortonCodes[i + 1]);
    int l, r, mid;
    if (dl > dr) {
        l = -1;
        r = i - 1;
        while (r - l > 1) {
            mid = (l + r) >> 1;
            commonUpperBits(code, mortonCodes[mid]) > dr ? r = mid : l = mid;
        }
        left = r;
        right = i;
    } else {
        l = i + 1;
        r = nNodes + 1;
        while (r - l > 1) {
            mid = (l + r) >> 1;
            commonUpperBits(code, mortonCodes[mid]) > dl ? l = mid : r = mid;
        }
        left = i;
        right = l;
    }
}

__device__ int findSplit(const unsigned long long* mortonCodes, int left, int right) {
    unsigned long long code = mortonCodes[left];
    int d = commonUpperBits(code, mortonCodes[right]);
    int l = left, r = right, mid;
    while (r - l > 1) {
        mid = (l + r) >> 1;
        commonUpperBits(code, mortonCodes[mid]) > d ? l = mid : r = mid;
    }
    return l;
}

__global__ void initializeInternalNodes(int nNodes, const unsigned long long* mortonCodes, BVHNode* leaves, BVHNode* internals) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads) {
        BVHNode& node = internals[i];
        int left, right;
        findRange(nNodes, mortonCodes, i, left, right);
        int middle = findSplit(mortonCodes, left, right);
        node.left = middle == left ? &leaves[middle] : &internals[middle];
        node.right = middle + 1 == right ? &leaves[middle + 1] : &internals[middle + 1];
        node.left->parent = node.right->parent = &node;
    }
}

__device__ float atomicMin(float* address, float val) {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed, __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ float atomicMax(float* address, float val) {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed, __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void computeInternalBounds(int nNodes, BVHNode* nodes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads) {
        nodes[i].maxIndex = i;
        const Bounds& bounds = nodes[i].bounds;
        BVHNode* node = nodes[i].parent;

        while (node != nullptr) {
            atomicMax(&node->maxIndex, i);
            for (int j = 0; j < 3; j++) {
                atomicMin(&node->bounds.pMin(j), bounds.pMin(j));
                atomicMax(&node->bounds.pMax(j), bounds.pMax(j));
            }
            node = node->parent;
        }
        // while (node != nullptr)
        //     if (atomicCAS(&node->count, 0, 1) == 1) {
        //         node->bounds = node->left->bounds + node->right->bounds;
        //         node->maxIndex = max(node->left->maxIndex, node->right->maxIndex);
        //         node = node->parent;
        //     } else
        //         node = nullptr;
    }
}

__global__ void countPairsSelf(int nLeaves, const BVHNode* leaves, const BVHNode* root, float thickness, int* num) {
    int nThreads = gridDim.x * blockDim.x;

    const BVHNode* stack[64];
    stack[0] = nullptr;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nLeaves; i += nThreads) {
        const Bounds& bounds = leaves[i].bounds;
        int& n = num[i];
        n = 0;
        int top = 0;
        const BVHNode* node = root;

        do {
            const BVHNode* left = node->left;
            const BVHNode* right = node->right;

            bool overlapLeft = i < left->maxIndex && bounds.overlap(left->bounds, thickness);
            bool overlapRight = i < right->maxIndex && bounds.overlap(right->bounds, thickness);
            if (overlapLeft && left->isLeaf())
                n++;
            if (overlapRight && right->isLeaf())
                n++;

            bool traverseLeft = (overlapLeft && !left->isLeaf());
            bool traverseRight = (overlapRight && !right->isLeaf());
            if (!traverseLeft && !traverseRight)
                node = stack[top--];
            else {
                node = traverseLeft ? left : right;
                if (traverseLeft && traverseRight)
                    stack[++top] = right;
            }
        } while (node != nullptr);
    }
}

__global__ void findPairsSelf(int nLeaves, const BVHNode* leaves, const BVHNode* root, float thickness, const int* num, PairFF* pairs) {
    int nThreads = gridDim.x * blockDim.x;

    const BVHNode* stack[64];
    stack[0] = nullptr;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nLeaves; i += nThreads) {
        const BVHNode& leaf = leaves[i];
        Face* face =  leaf.face;
        const Bounds& bounds = leaf.bounds;
        int index = num[i] - 1;
        int top = 0;
        const BVHNode* node = root;

        do {
            const BVHNode* left = node->left;
            const BVHNode* right = node->right;

            bool overlapLeft = i < left->maxIndex && bounds.overlap(left->bounds, thickness);
            bool overlapRight = i < right->maxIndex && bounds.overlap(right->bounds, thickness);
            if (overlapLeft && left->isLeaf()) {
                PairFF& pair = pairs[index--];
                pair.first = face;
                pair.second = left->face;
            }
            if (overlapRight && right->isLeaf()) {
                PairFF& pair = pairs[index--];
                pair.first = face;
                pair.second = right->face;
            }

            bool traverseLeft = (overlapLeft && !left->isLeaf());
            bool traverseRight = (overlapRight && !right->isLeaf());
            if (!traverseLeft && !traverseRight)
                node = stack[top--];
            else {
                node = traverseLeft ? left : right;
                if (traverseLeft && traverseRight)
                    stack[++top] = right;
            }
        } while (node != nullptr);
    }
}

__global__ void countPairs(int nLeaves, const BVHNode* leaves, const BVHNode* root, float thickness, int* num) {
    int nThreads = gridDim.x * blockDim.x;

    const BVHNode* stack[64];
    stack[0] = nullptr;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nLeaves; i += nThreads) {
        const Bounds& bounds = leaves[i].bounds;
        int& n = num[i];
        n = 0;
        int top = 0;
        const BVHNode* node = root;

        do {
            const BVHNode* left = node->left;
            const BVHNode* right = node->right;

            bool overlapLeft = bounds.overlap(left->bounds, thickness);
            bool overlapRight = bounds.overlap(right->bounds, thickness);
            if (overlapLeft && left->isLeaf())
                n++;
            if (overlapRight && right->isLeaf())
                n++;

            bool traverseLeft = (overlapLeft && !left->isLeaf());
            bool traverseRight = (overlapRight && !right->isLeaf());
            if (!traverseLeft && !traverseRight)
                node = stack[top--];
            else {
                node = traverseLeft ? left : right;
                if (traverseLeft && traverseRight)
                    stack[++top] = right;
            }
        } while (node != nullptr);
    }
}

__global__ void findPairs(int nLeaves, const BVHNode* leaves, const BVHNode* root, float thickness, const int* num, PairFF* pairs) {
    int nThreads = gridDim.x * blockDim.x;

    const BVHNode* stack[64];
    stack[0] = nullptr;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nLeaves; i += nThreads) {
        const BVHNode& leaf = leaves[i];
        Face* face =  leaf.face;
        const Bounds& bounds = leaf.bounds;
        int index = num[i] - 1;
        int top = 0;
        const BVHNode* node = root;

        do {
            const BVHNode* left = node->left;
            const BVHNode* right = node->right;

            bool overlapLeft = bounds.overlap(left->bounds, thickness);
            bool overlapRight = bounds.overlap(right->bounds, thickness);
            if (overlapLeft && left->isLeaf()) {
                PairFF& pair = pairs[index--];
                pair.first = face;
                pair.second = left->face;
            }
            if (overlapRight && right->isLeaf()) {
                PairFF& pair = pairs[index--];
                pair.first = face;
                pair.second = right->face;
            }

            bool traverseLeft = (overlapLeft && !left->isLeaf());
            bool traverseRight = (overlapRight && !right->isLeaf());
            if (!traverseLeft && !traverseRight)
                node = stack[top--];
            else {
                node = traverseLeft ? left : right;
                if (traverseLeft && traverseRight)
                    stack[++top] = right;
            }
        } while (node != nullptr);
    }
}

__global__ void findNearestPointGpu(int nNodes, const Vector3f* x, const BVHNode* root, NearPoint* points) {
    int nThreads = gridDim.x * blockDim.x;

    const BVHNode* stack[64];
    stack[0] = nullptr;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads) {
        Vector3f xt = x[i];
        NearPoint& point = points[i];
        int top = 0;
        const BVHNode* node = root;

        do {
            const BVHNode* left = node->left;
            const BVHNode* right = node->right;

            bool overlapLeft = left->bounds.distance(xt) < point.d;
            bool overlapRight = right->bounds.distance(xt) < point.d;
            if (overlapLeft && left->isLeaf())
                checkNearestPoint(xt, left->face, point);
            if (overlapRight && right->isLeaf())
                checkNearestPoint(xt, right->face, point);

            bool traverseLeft = (overlapLeft && !left->isLeaf());
            bool traverseRight = (overlapRight && !right->isLeaf());
            if (!traverseLeft && !traverseRight)
                node = stack[top--];
            else {
                node = traverseLeft ? left : right;
                if (traverseLeft && traverseRight)
                    stack[++top] = right;
            }
        } while (node != nullptr);
    }
}

__global__ void resetCount(int nNodes, BVHNode* nodes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads)
        nodes[i].count = 0;
}

__global__ void updateGpu(int nNodes, BVHNode* nodes, bool ccd) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads) {
        nodes[i].bounds = nodes[i].face->bounds(ccd);
        const Bounds& bounds = nodes[i].bounds;
        BVHNode* node = nodes[i].parent;

        while (node != nullptr) {
            for (int j = 0; j < 3; j++) {
                atomicMin(&node->bounds.pMin(j), bounds.pMin(j));
                atomicMax(&node->bounds.pMax(j), bounds.pMax(j));
            }
            node = node->parent;
        }
        // while (node != nullptr)
        //     if (atomicCAS(&node->count, 0, 1) == 1) {
        //         node->bounds = node->left->bounds + node->right->bounds;
        //         node = node->parent;
        //     } else
        //         break;
    }
}