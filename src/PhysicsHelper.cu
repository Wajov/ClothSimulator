#include "PhysicsHelper.cuh"

bool inEdge(float w, const Edge* edge0, const Edge* edge1) {
    Vector3f x = (1.0f - w) * edge0->nodes[0]->x + w * edge0->nodes[1]->x;
    bool in = true;
    for (int i = 0; i < 2; i++) {
        Face* face = edge1->adjacents[i];
        if (face == nullptr)
            continue;
        Node* node0 = edge1->nodes[i];
        Node* node1 = edge1->nodes[1 - i];
        Vector3f e = node1->x - node0->x;
        Vector3f n = face->n;
        Vector3f r = x - node0->x;
        in &= (mixed(e, n, r) >= 0.0f);
    }
    return in;
}

__global__ void addMass(int nNodes, const Node* const* nodes, Pairii* aIndices, float* aValues) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads) {
        float mass = nodes[i]->mass;
        for (int j = 0; j < 3; j++) {
            int index = 3 * i + j;
            aIndices[index] = Pairii(index, index);
            aValues[index] = mass;
        }
    }
}

__global__ void addGravity(int nNodes, const Node* const* nodes, float dt, const Vector3f gravity, int* bIndices, float* bValues) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads) {
        Vector3f g = dt * nodes[i]->mass * gravity;
        for (int j = 0; j < 3; j++) {
            int index = 3 * i + j;
            bIndices[index] = index;
            bValues[index] = g(j);
        }
    }
}

__global__ void addWindForces(int nFaces, const Face* const* faces, float dt, const Wind* wind, int* bIndices, float* bValues) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads) {
        const Face* face = faces[i];
        Vector3f n = face->n;
        float area = face->area;
        Vector3f average = (face->vertices[0]->node->v + face->vertices[1]->node->v + face->vertices[2]->node->v) / 3.0f;
        Vector3f relative = wind->getVelocity() - average;
        float vn = n.dot(relative);
        Vector3f vt = relative - vn * n;
        Vector3f force = area * (wind->getDensity() * abs(vn) * vn * n + wind->getDrag() * vt) / 3.0f;
        Vector3f f = dt * force;
        for (int j = 0; j < 3; j++) {
            int nodeIndex = face->vertices[j]->node->index;
            for (int k = 0; k < 3; k++) {
                int index = 9 * i + 3 * j + k;
                bIndices[index] = 3 * nodeIndex + k;
                bValues[index] = f(k);
            }
        }
    }
}

void stretchingForce(const Face* face, const Material* material, Vector9f& f, Matrix9x9f& J) {
    Matrix3x2f F = face->derivative(face->vertices[0]->node->x, face->vertices[1]->node->x, face->vertices[2]->node->x);
    Matrix2x2f G = 0.5f * (F.transpose() * F - Matrix2x2f(1.0f));

    Matrix2x2f Y = face->inverse;
    Matrix2x3f D(-Y.row(0) - Y.row(1), Y.row(0), Y.row(1));
    Matrix3x9f Du(Matrix3x3f(D(0, 0)), Matrix3x3f(D(0, 1)), Matrix3x3f(D(0, 2)));
    Matrix3x9f Dv(Matrix3x3f(D(1, 0)), Matrix3x3f(D(1, 1)), Matrix3x3f(D(1, 2)));

    Vector3f fu = F.col(0);
    Vector3f fv = F.col(1);

    Vector9f fuu = Du.transpose() * fu;
    Vector9f fvv = Dv.transpose() * fv;
    Vector9f fuv = 0.5f * (Du.transpose() * fv + Dv.transpose() * fu);

    Vector4f k = material->stretchingStiffness(G);

    Vector9f grad = k(0) * G(0, 0) * fuu + k(2) * G(1, 1) * fvv + k(1) * (G(0, 0) * fvv + G(1, 1) * fuu) + 2.0f * k(3) * G(0, 1) * fuv;
    Matrix9x9f hess = k(0) * (fuu.outer(fuu) + max(G(0, 0), 0.0f) * Du.transpose() * Du)
                    + k(2) * (fvv.outer(fvv) + max(G(1, 1), 0.0f) * Dv.transpose() * Dv)
                    + k(1) * (fuu.outer(fvv) + max(G(0, 0), 0.0f) * Dv.transpose() * Dv + fvv.outer(fuu) + max(G(1, 1), 0.0f) * Du.transpose() * Du)
                    + 2.0f * k(3) * fuv.outer(fuv);

    float area = face->area;
    f = -area * grad;
    J = -area * hess;
}

__global__ void addStretchingForces(int nFaces, const Face* const* faces, float dt, const Material* material, Pairii* aIndices, float* aValues, int* bIndices, float* bValues) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads) {
        const Face* face = faces[i];
        Node* node0 = face->vertices[0]->node;
        Node* node1 = face->vertices[1]->node;
        Node* node2 = face->vertices[2]->node;
        Vector9f v(node0->v, node1->v, node2->v);

        Vector9f f;
        Matrix9x9f J;
        stretchingForce(face, material, f, J);
        
        f = dt * (f + dt * J * v);
        J = -dt * dt * J;
        Vector3i indices(node0->index, node1->index, node2->index);
        
        for (int j = 0; j < 3; j++) {
            int x = indices(j);
            for (int k = 0; k < 3; k++) {
                int y = indices(k);
                for (int l = 0; l < 3; l++)
                    for (int r = 0; r < 3; r++) {
                        int index = 81 * i + 27 * j + 9 * k + 3 * l + r;
                        aIndices[index] = Pairii(3 * x + l, 3 * y + r);
                        aValues[index] = J(3 * j + l, 3 * k + r);
                    }
            }
        }

        for (int j = 0; j < 3; j++) {
            int x = indices(j);
            for (int k = 0; k < 3; k++) {
                int index = 9 * i + 3 * j + k;
                bIndices[index] = 3 * x + k;
                bValues[index] = f(3 * j + k);
            }
        }
    }
}

float distance(const Vector3f& x, const Vector3f& a, const Vector3f& b) {
    Vector3f e = b - a;
    Vector3f t = x - a;
    Vector3f r = e * e.dot(t) / e.norm2();
    return (t - r).norm();
}

Vector2f barycentricWeights(const Vector3f& x, const Vector3f& a, const Vector3f& b) {
    Vector3f e = b - a;
    float t = e.dot(x - a) / e.norm2();
    return Vector2f(1.0f - t, t);
}

void bendingForce(const Edge* edge, const Material* material, Vector12f& f, Matrix12x12f& J) {
    Vector3f x0 = edge->nodes[0]->x;
    Vector3f x1 = edge->nodes[1]->x;
    Vector3f x2 = edge->opposites[0]->node->x;
    Vector3f x3 = edge->opposites[1]->node->x;
    Face* adjacent0 = edge->adjacents[0];
    Face* adjacent1 = edge->adjacents[1];
    Vector3f n0 = adjacent0->n;
    Vector3f n1 = adjacent1->n;
    float length = edge->length();
    float angle = edge->angle();
    float area = adjacent0->area + adjacent1->area;

    float h0 = distance(x2, x0, x1);
    float h1 = distance(x3, x0, x1);
    Vector2f w0 = barycentricWeights(x2, x0, x1);
    Vector2f w1 = barycentricWeights(x3, x0, x1);

    Vector12f dtheta(-w0(0) * n0 / h0 - w1(0) * n1 / h1, -w0(1) * n0 / h0 - w1(1) * n1 / h1, n0 / h0, n1 / h1);

    float k0 = material->bendingStiffness(length, angle, area, edge->vertices[0][1]->u - edge->vertices[0][0]->u);
    float k1 = material->bendingStiffness(length, angle, area, edge->vertices[1][1]->u - edge->vertices[1][0]->u);
    float coefficient = -0.25f * min(k0, k1) * sqr(length) / area;

    f = coefficient * angle * dtheta;
    J = coefficient * dtheta.outer(dtheta);
}

__global__ void addBendingForces(int nEdges, const Edge* const* edges, float dt, const Material* material, Pairii* aIndices, float* aValues, int* bIndices, float* bValues) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads) {
        const Edge* edge = edges[i];
        Vector12f f;
        Matrix12x12f J;
        Vector4i indices;

        if (!edge->isBoundary()) {
            Node* node0 = edge->nodes[0];
            Node* node1 = edge->nodes[1];
            Node* node2 = edge->opposites[0]->node;
            Node* node3 = edge->opposites[1]->node;
            Vector12f v(node0->v, node1->v, node2->v, node3->v);

            bendingForce(edge, material, f, J);
            f = dt * (f + dt * J * v);
            J = -dt * dt * J;
            indices(0) = node0->index;
            indices(1) = node1->index;
            indices(2) = node2->index;
            indices(3) = node3->index;
        }
        
        for (int j = 0; j < 4; j++) {
            int x = indices(j);
            for (int k = 0; k < 4; k++) {
                int y = indices(k);
                for (int l = 0; l < 3; l++)
                    for (int r = 0; r < 3; r++) {
                        int index = 144 * i + 36 * j + 9 * k + 3 * l + r;
                        aIndices[index] = Pairii(3 * x + l, 3 * y + r);
                        aValues[index] = J(3 * j + l, 3 * k + r);
                    }
            }
        }

        for (int j = 0; j < 4; j++) {
            int x = indices(j);
            for (int k = 0; k < 3; k++) {
                int index = 12 * i + 3 * j + k;
                bIndices[index] = 3 * x + k;
                bValues[index] = f(3 * j + k);
            }
        }
    }
}

__global__ void addHandleForcesGpu(int nHandles, const Handle* handles, float dt, float stiffness, Pairii* aIndices, float* aValues, int* bIndices, float* bValues) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nHandles; i += nThreads) {
        const Handle& handle = handles[i];
        Node* node = handle.node;
        int nodeIndex = node->index;
        Vector3f f = dt * ((handle.position - node->x) - dt * node->v) * stiffness;
        for (int j = 0; j < 3; j++) {
            int index = 3 * i + j;
            aIndices[index] = Pairii(3 * nodeIndex + j, 3 * nodeIndex + j);
            aValues[index] = dt * dt * stiffness;
            bIndices[index] = 3 * nodeIndex + j;
            bValues[index] = f(j);
        }
    }
}

__host__ __device__ void impulseForce(const Proximity& proximity, float thickness, Vector12f& f, Matrix12x12f& J) {
    Node* const* nodes = proximity.nodes;
    float const* w = proximity.w;
    Vector3f n = proximity.n;
    float stiffness = proximity.stiffness;
    float d = -thickness;
    for (int i = 0; i < 4; i++)
        d += w[i] * n.dot(nodes[i]->x);
    d = max(-d, 0.0f);
    Vector12f N(w[0] * n, w[1] * n, w[2] * n, w[3] * n);
    f = 0.5f * stiffness / thickness * sqr(d) * N;
    J = -stiffness / thickness * d * N.outer(N);
}

__global__ void splitIndices(int nIndices, const Pairii* indices, int* rowIndices, int* colIndices) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nIndices; i += nThreads) {
        rowIndices[i] = indices[i].first;
        colIndices[i] = indices[i].second;
    }
}

__global__ void setVector(int nIndices, const int* indices, const float* values, float* v) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nIndices; i += nThreads)
        v[indices[i]] = values[i];
}

__global__ void updateNodes(int nNodes, float dt, const float* dv, Node** nodes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads)
        nodes[i]->v += Vector3f(dv[3 * i], dv[3 * i + 1], dv[3 * i + 2]);
}