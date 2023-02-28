#include "CollisionHelper.cuh"

float newtonsMethod(float a, float b, float c, float d, float x0, int dir) {
    if (dir != 0) {
        float y0 = d + x0 * (c + x0 * (b + x0 * a));
        float ddy0 = 2.0f * b + x0 * (6.0f * a);
        x0 += dir * sqrt(abs(2.0f * y0 / ddy0));
    }
    for (int iter = 0; iter < 100; iter++) {
        float y = d + x0 * (c + x0 * (b + x0 * a));
        float dy = c + x0 * (2*b + 3.0f * x0 * a);
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
    if (abs(a) > 1e-12 * abs(q))
        x[i++] = q / a;
    if (abs(q) > 1e-12 * abs(c))
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

float signedVertexFaceDistance(const Vector3f& x, const Vector3f& y0, const Vector3f& y1, const Vector3f& y2, Vector3f& n, float* w) {
    n = (y1 - y0).normalized().cross((y2 - y0).normalized());
    if (n.norm2() < 1e-6f)
        return INFINITY;
    n.normalize();
    float h = (x - y0).dot(n);
    float b0 = mixed(y1 - x, y2 - x, n);
    float b1 = mixed(y2 - x, y0 - x, n);
    float b2 = mixed(y0 - x, y1 - x, n);
    w[0] = 1.0f;
    w[1] = -b0 / (b0 + b1 + b2);
    w[2] = -b1 / (b0 + b1 + b2);
    w[3] = -b2 / (b0 + b1 + b2);
    return h;
}

float signedEdgeEdgeDistance(const Vector3f& x0, const Vector3f& x1, const Vector3f& y0, const Vector3f& y1, Vector3f& n, float* w) {
    n = (x1 - x0).normalized().cross((y1 - y0).normalized());
    if (n.norm2() < 1e-8f) {
        Vector3f e0 = (x1 - x0).normalized(), e1 = (y1 - y0).normalized();
        float p0min = x0.dot(e0), p0max = x1.dot(e0), p1min = y0.dot(e0), p1max = y1.dot(e0);
        if (p1max < p1min)
            mySwap(p1max, p1min);
        
        float a = max(p0min, p1min), b = min(p0max, p1max), c = 0.5f * (a + b);
        if (a > b)
            return INFINITY;
        
        Vector3f d = y0 - x0 - (y0-x0).dot(e0) * e0;
        n = (-d).normalized();
        w[1] = (c - x0.dot(e0)) / (x1 - x0).norm();
        w[0] = 1.0f - w[1];
        w[3] = -(e0.dot(e1) * c - y0.dot(e1)) / (y1-y0).norm();
        w[2] = -1.0f - w[3];
        return d.norm();
    }
    n = n.normalized();
    float h = (x0 - y0).dot(n);
    float a0 = mixed(y1 - x1, y0 - x1, n);
    float a1 = mixed(y0 - x0, y1 - x0, n);
    float b0 = mixed(x0 - y1, x1 - y1, n);
    float b1 = mixed(x1 - y0, x0 - y0, n);
    w[0] = a0 / (a0 + a1);
    w[1] = a1 / (a0 + a1);
    w[2] = -b0 / (b0 + b1);
    w[3] = -b1 / (b0 + b1);
    return h;
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

    if (abs(a0) < 1e-6f * x1.norm() * x2.norm() * x3.norm())
        return false;

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

__global__ void checkImpactsGpu(int nProximities, const Proximity* proximities, float thickness, Impact* impacts) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nProximities; i += nThreads) {
        const Proximity& proximity = proximities[i];
        const Face* face0 = proximity.first;
        const Face* face1 = proximity.second;
        int n = 15 * i;
        for (int j = 0; j < 3; j++) {
            Impact& impact = impacts[n++];
            if (!checkVertexFaceImpact(face0->vertices[j], face1, thickness, impact))
                impact.t = -1.0f;
        }
        for (int j = 0; j < 3; j++) {
            Impact& impact = impacts[n++];
            if (!checkVertexFaceImpact(face1->vertices[j], face0, thickness, impact))
                impact.t = -1.0f;
        }
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++) {
                Impact& impact = impacts[n++];
                if (!checkEdgeEdgeImpact(face0->edges[j], face1->edges[k], thickness, impact))
                    impact.t = -1.0f;
            }
    }
}

__global__ void collectNodeImpacts(int nImpacts, const Impact* impacts, Node** nodes, Pairfi* nodeImpacts) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nImpacts; i += nThreads) {
        const Impact& impact = impacts[i];
        for (int j = 0; j < 4; j++) {
            int index = 4 * i + j;
            nodes[index] = impact.nodes[j];
            nodeImpacts[index] = Pairfi(impact.t, i);
        }
    }
}

__global__ void setIndependentImpacts(int nImpacts, const Pairfi* nodeImpacts, const Impact* impacts, Impact* independentImpacts) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nImpacts; i += nThreads)
        independentImpacts[i] = impacts[nodeImpacts[i].second];
}
