#include "CollisionHelper.cuh"

float newtonsMethod(float a, float b, float c, float d, float x0, int dir) {
    if (dir != 0) {
        float y0 = d + x0 * (c + x0 * (b + x0 * a));
        float ddy0 = 2.0f * b + x0 * (6.0f * a);
        x0 += dir * sqrt(abs(2.0f * y0 / ddy0));
    }
    for (int iter = 0; iter < 100; iter++) {
        float y = d + x0 * (c + x0 * (b + x0 * a));
        float dy = c + x0 * (2.0f * b + 3.0f * x0 * a);
        if (dy == 0)
            return x0;
        float x1 = x0 - y / dy;
        if (abs(x0 - x1) < 1e-6f)
            return x0;
        x0 = x1;
    }
    return x0;
}

int solveQuadratic(float a, float b, float c, float x[2]) {
    float d = b * b - 4.0f * a * c;
    if (d < 0.0f) {
        x[0] = -b / (2.0f * a);
        return 0;
    }
    float q = -(b + sign(b) * sqrt(d)) * 0.5f;
    int i = 0;
    if (abs(a) > 1e-12f * abs(q))
        x[i++] = q / a;
    if (abs(q) > 1e-12f * abs(c))
        x[i++] = c / q;
    if (i == 2 && x[0] > x[1])
        mySwap(x[0], x[1]);
    return i;
}

int solveCubic(float a, float b, float c, float d, float x[]) {
    float xc[2];
    int n = solveQuadratic(3.0f * a, 2.0f * b, c, xc);
    if (n == 0) {
        x[0] = newtonsMethod(a, b, c, d, xc[0], 0);
        return 1;
    } else if (n == 1)
        return solveQuadratic(b, c, d, x);
    else {
        float yc[2] = {d + xc[0] * (c + xc[0] * (b + xc[0] * a)), d + xc[1] * (c + xc[1] * (b + xc[1] * a))};
        int i = 0;
        if (yc[0] * a >= 0.0f)
            x[i++] = newtonsMethod(a, b, c, d, xc[0], -1);
        if (yc[0] * yc[1] <= 0.0f) {
            int closer = abs(yc[0]) < abs(yc[1]) ? 0 : 1;
            x[i++] = newtonsMethod(a, b, c, d, xc[closer], closer == 0 ? 1 : -1);
        }
        if (yc[1] * a <= 0.0f)
            x[i++] = newtonsMethod(a, b, c, d, xc[1], 1);
        return i;
    }
}

bool checkImpact(ImpactType type, const Node* node0, const Node* node1, const Node* node2, const Node* node3, Impact& impact) {
    impact.nodes[0] = const_cast<Node*>(node0);
    impact.nodes[1] = const_cast<Node*>(node1);
    impact.nodes[2] = const_cast<Node*>(node2);
    impact.nodes[3] = const_cast<Node*>(node3);

    Vector3f x0 = node0->x0;
    Vector3f v0 = node0->x - x0;
    Vector3f x1 = node1->x0 - x0;
    Vector3f x2 = node2->x0 - x0;
    Vector3f x3 = node3->x0 - x0;
    Vector3f v1 = (node1->x - node1->x0) - v0;
    Vector3f v2 = (node2->x - node2->x0) - v0;
    Vector3f v3 = (node3->x - node3->x0) - v0;
    float a0 = mixed(x1, x2, x3);
    float a1 = mixed(v1, x2, x3) + mixed(x1, v2, x3) + mixed(x1, x2, v3);
    float a2 = mixed(x1, v2, v3) + mixed(v1, x2, v3) + mixed(v1, v2, x3);
    float a3 = mixed(v1, v2, v3);

    float t[3];
    int nSolution = solveCubic(a3, a2, a1, a0, t);
    for (int i = 0; i < nSolution; i++) {
        if (t[i] < 0.0f || t[i] > 1.0f)
            continue;
        impact.t = t[i];
        Vector3f x0 = node0->position(t[i]);
        Vector3f x1 = node1->position(t[i]);
        Vector3f x2 = node2->position(t[i]);
        Vector3f x3 = node3->position(t[i]);

        Vector3f& n = impact.n;
        float* w = impact.w;
        float d;
        bool inside;
        if (type == VertexFace) {
            d = signedVertexFaceDistance(x0, x1, x2, x3, n, w);
            inside = (min(-w[1], -w[2], -w[3]) >= -1e-6f);
        } else {
            d = signedEdgeEdgeDistance(x0, x1, x2, x3, n, w);
            inside = (min(w[0], w[1], -w[2], -w[3]) >= -1e-6f);
        }
        if (n.dot(w[1] * v1 + w[2] * v2 + w[3] * v3) > 0.0f)
            n = -n;
        if (abs(d) < 1e-6f && inside)
            return true;
    }
    return false;
}

bool checkVertexFaceImpact(const Vertex* vertex, const Face* face, float thickness, Impact& impact) {
    Node* node = vertex->node;
    Node* node0 = face->vertices[0]->node;
    Node* node1 = face->vertices[1]->node;
    Node* node2 = face->vertices[2]->node;
    if (node == node0 || node == node1 || node == node2)
        return false;
    if (!node->bounds(true).overlap(face->bounds(true), thickness))
        return false;
    
    return checkImpact(VertexFace, node, node0, node1, node2, impact);
}

bool checkEdgeEdgeImpact(const Edge* edge0, const Edge* edge1, float thickness, Impact& impact) {
    Node* node0 = edge0->nodes[0];
    Node* node1 = edge0->nodes[1];
    Node* node2 = edge1->nodes[0];
    Node* node3 = edge1->nodes[1];
    if (node0 == node2 || node0 == node3 || node1 == node2 || node1 == node3)
        return false;
    if (!edge0->bounds(true).overlap(edge1->bounds(true), thickness))
        return false;
    
    return checkImpact(EdgeEdge, node0, node1, node2, node3, impact);
}

__global__ void checkImpactsGpu(int nPairs, const PairFF* pairs, float thickness, Impact* impacts) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nPairs; i += nThreads) {
        const PairFF& pair = pairs[i];
        Face* face0 = pair.first;
        Face* face1 = pair.second;
        int index = 15 * i;
        for (int j = 0; j < 3; j++) {
            Impact& impact = impacts[index++];
            if (!checkVertexFaceImpact(face0->vertices[j], face1, thickness, impact))
                impact.t = -1.0f;
        }
        for (int j = 0; j < 3; j++) {
            Impact& impact = impacts[index++];
            if (!checkVertexFaceImpact(face1->vertices[j], face0, thickness, impact))
                impact.t = -1.0f;
        }
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++) {
                Impact& impact = impacts[index++];
                if (!checkEdgeEdgeImpact(face0->edges[j], face1->edges[k], thickness, impact))
                    impact.t = -1.0f;
            }
    }
}

__global__ void initializeImpactNodes(int nImpacts, const Impact* impacts, int deform) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nImpacts; i += nThreads) {
        const Impact& impact = impacts[i];
        for (int j = 0; j < 4; j++) {
            Node* node = impact.nodes[j];
            if (deform == 1 || node->isFree) {
                node->removed = false;
                node->minIndex = nImpacts;
            }
        }
    }
}

__global__ void collectRelativeImpacts(int nImpacts, const Impact* impacts, int deform, Node** nodes, int* relativeImpacts) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nImpacts; i += nThreads) {
        const Impact& impact = impacts[i];
        bool flag = true;
        for (int j = 0; j < 4; j++) {
            Node* node = impact.nodes[j];
            if ((deform == 1 || node->isFree) && node->removed) {
                flag = false;
                break;
            }
        }

        for (int j = 0; j < 4; j++) {
            Node* node = impact.nodes[j];
            int index = 4 * i + j;
            if (flag && (deform == 1 || node->isFree)) {
                nodes[index] = node;
                relativeImpacts[index] = i;
            }  else {
                nodes[index] = nullptr;
                relativeImpacts[index] = -1;
            }
        }
    }
}

__global__ void setImpactMinIndices(int nNodes, const int* relativeImpacts, Node** nodes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads)
        nodes[i]->minIndex = relativeImpacts[i];
}

__global__ void checkIndependentImpacts(int nImpacts, const Impact* impacts, int deform, Impact* independentImpacts) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nImpacts; i += nThreads) {
        const Impact& impact = impacts[i];
        bool flag = true;
        for (int j = 0; j < 4; j++) {
            Node* node = impact.nodes[j];
            if ((deform == 1 || node->isFree) && (node->removed || node->minIndex != i)) {
                flag = false;
                break;
            }
        }

        if (flag) {
            independentImpacts[i] = impact;
            for (int j = 0; j < 4; j++) {
                Node* node = impact.nodes[j];
                if (deform == 1 || node->isFree)
                    node->removed = true;
            }
        }
    }
}