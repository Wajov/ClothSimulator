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

__device__ void checkVertexFaceProximityGpu(const Vertex* vertex, const Face* face, PairNi& key0, PairfF& value0, PairFi& key1, PairfN& value1) {
    Node* node = vertex->node;
    Node* node0 = face->vertices[0]->node;
    Node* node1 = face->vertices[1]->node;
    Node* node2 = face->vertices[2]->node;
    if (node == node0 || node == node1 || node == node2) {
        key0.first = nullptr;
        key1.first = nullptr;
        return;
    }
    
    Vector3f n;
    float w[4];
    float d = abs(signedVertexFaceDistance(node->x, node0->x, node1->x, node2->x, n, w));
    bool inside = (min(-w[1], -w[2], -w[3]) >= 1e-6f);
    if (!inside) {
        key0.first = nullptr;
        key1.first = nullptr;
        return;
    }

    if (inside && node->isFree) {
        int side = n.dot(node->n) >= 0.0f ? 0 : 1;
        key0 = PairNi(node, side);
        value0 = PairfF(d, const_cast<Face*>(face));
    } else
        key0.first = nullptr;
    if (face->isFree()) {
        int side = -n.dot(face->n) >= 0.0f ? 0 : 1;
        key1 = PairFi(const_cast<Face*>(face), side);
        value1 = PairfN(d, node);
    } else
        key1.first = nullptr;
}

__device__ void checkEdgeEdgeProximityGpu(const Edge* edge0, const Edge* edge1, PairEi& key0, PairfE& value0, PairEi& key1, PairfE& value1) {
    Node* node0 = edge0->nodes[0];
    Node* node1 = edge0->nodes[1];
    Node* node2 = edge1->nodes[0];
    Node* node3 = edge1->nodes[1];
    if (node0 == node2 || node0 == node3 || node1 == node2 || node1 == node3) {
        key0.first = key1.first = nullptr;
        return;
    }
    
    Vector3f n;
    float w[4];
    float d = abs(signedEdgeEdgeDistance(node0->x, node1->x, node2->x, node3->x, n, w));
    bool inside = (min(w[0], w[1], -w[2], -w[3]) >= 1e-6f && inEdge(w[1], edge0, edge1) && inEdge(-w[3], edge1, edge0));
    if (!inside) {
        key0.first = key1.first = nullptr;
        return;
    }
    
    if (edge0->isFree()) {
        int side = n.dot(edge0->nodes[0]->n + edge0->nodes[1]->n) >= 0.0f ? 0 : 1;
        key0 = PairEi(const_cast<Edge*>(edge0), side);
        value0 = PairfE(d, const_cast<Edge*>(edge1));
    } else
        key0.first = nullptr;
    if (edge1->isFree()) {
        int side = -n.dot(edge1->nodes[0]->n + edge1->nodes[1]->n) >= 0.0f ? 0 : 1;
        key1 = PairEi(const_cast<Edge*>(edge1), side);
        value1 = PairfE(d, const_cast<Edge*>(edge0));
    } else
        key1.first = nullptr;
}

__global__ void checkProximitiesGpu(int nPairs, const PairFF* pairs, PairNi* nodes, PairfF* nodeProximities, PairEi* edges, PairfE* edgeProximities, PairFi* faces, PairfN* faceProximities) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nPairs; i += nThreads) {
        const PairFF& pair = pairs[i];
        Face* face0 = pair.first;
        Face* face1 = pair.second;
        int index0 = 6 * i;
        int index1 = 18 * i;
        for (int j = 0; j < 3; j++) {
            checkVertexFaceProximityGpu(face0->vertices[j], face1, nodes[index0], nodeProximities[index0], faces[index0], faceProximities[index0]);
            index0++;
        }
        for (int j = 0; j < 3; j++) {
            checkVertexFaceProximityGpu(face1->vertices[j], face0, nodes[index0], nodeProximities[index0], faces[index0], faceProximities[index0]);
            index0++;
        }
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++) {
                checkEdgeEdgeProximityGpu(face0->edges[j], face1->edges[k], edges[index1], edgeProximities[index1], edges[index1 + 1], edgeProximities[index1 + 1]);
                index1 += 2;
            }
    }
}

__global__ void setNodeProximities(int nProximities, const PairNi* nodes, const PairfF* nodeProximities, float thickness, float stiffness, float clothFriction, float obstacleFriction, Proximity* proximities) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nProximities; i += nThreads) {
        const PairNi& node = nodes[i];
        const PairfF& nodeProximity = nodeProximities[i];
        if (nodeProximity.first < 2.0f * thickness)
            proximities[i] = Proximity(node.first, nodeProximity.second, stiffness, clothFriction, obstacleFriction);
        else
            proximities[i].stiffness = -1.0f;
    }
}

__global__ void setEdgeProximities(int nProximities, const PairEi* edges, const PairfE* edgeProximities, float thickness, float stiffness, float clothFriction, float obstacleFriction, Proximity* proximities) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nProximities; i += nThreads) {
        const PairEi& edge = edges[i];
        const PairfE& edgeProximity = edgeProximities[i];
        if (edgeProximity.first < 2.0f * thickness)
            proximities[i] = Proximity(edge.first, edgeProximity.second, stiffness, clothFriction, obstacleFriction);
        else
            proximities[i].stiffness = -1.0f;
    }
}

__global__ void setFaceProximities(int nProximities, const PairFi* faces, const PairfN* faceProximities, float thickness, float stiffness, float clothFriction, float obstacleFriction, Proximity* proximities) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nProximities; i += nThreads) {
        const PairFi& face = faces[i];
        const PairfN& faceProximity = faceProximities[i];
        if (faceProximity.first < 2.0f * thickness)
            proximities[i] = Proximity(faceProximity.second, face.first, stiffness, clothFriction, obstacleFriction);
        else
            proximities[i].stiffness = -1.0f;
    }
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

__device__ void addMatrixAndVectorGpu(const Matrix9x9f& B, const Vector9f& b, const Vector3i& indices, Pairii* aIndices, float* aValues, int* bIndices, float* bValues) {
    for (int i = 0; i < 3; i++) {
        int x = indices(i);
        for (int j = 0; j < 3; j++) {
            int y = indices(j);
            for (int k = 0; k < 3; k++)
                for (int h = 0; h < 3; h++) {
                    int index = 27 * i + 9 * j + 3 * k + h;
                    if (x > -1 && y > -1) {
                        aIndices[index] = Pairii(3 * x + k, 3 * y + h);
                        aValues[index] = B(3 * i + k, 3 * j + h);
                    } else {
                        aIndices[index] = Pairii(0, 0);
                        aValues[index] = 0.0f;
                    }
                }
        }

        for (int j = 0; j < 3; j++) {
            int index = 3 * i + j;
            if (x > -1) {
                bIndices[index] = 3 * x + j;
                bValues[index] = b(3 * i + j);
            } else {
                bIndices[index] = 0;
                bValues[index] = 0.0f;
            }
        }
    }
}

__device__ void addMatrixAndVectorGpu(const Matrix12x12f& B, const Vector12f& b, const Vector4i& indices, Pairii* aIndices, float* aValues, int* bIndices, float* bValues) {
    for (int i = 0; i < 4; i++) {
        int x = indices(i);
        for (int j = 0; j < 4; j++) {
            int y = indices(j);
            for (int k = 0; k < 3; k++)
                for (int h = 0; h < 3; h++) {
                    int index = 36 * i + 9 * j + 3 * k + h;
                    if (x > -1 && y > -1) {
                        aIndices[index] = Pairii(3 * x + k, 3 * y + h);
                        aValues[index] = B(3 * i + k, 3 * j + h);
                    } else {
                        aIndices[index] = Pairii(0, 0);
                        aValues[index] = 0.0f;
                    }
                }
        }

        for (int j = 0; j < 3; j++) {
            int index = 3 * i + j;
            if (x > -1) {
                bIndices[index] = 3 * x + j;
                bValues[index] = b(3 * i + j);
            } else {
                bIndices[index] = 0;
                bValues[index] = 0.0f;
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
        Vector3i indices(node0->index, node1->index, node2->index);

        Vector9f f;
        Matrix9x9f J;
        stretchingForce(face, material, f, J);
        
        int aIndex = 81 * i;
        int bIndex = 9 * i;
        addMatrixAndVectorGpu(-dt * dt * J, dt * (f + dt * J * v), indices, aIndices + aIndex, aValues + aIndex, bIndices + bIndex, bValues + bIndex);
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
        int aIndex = 144 * i;
        int bIndex = 12 * i;
        if (!edge->isBoundary()) {
            Node* node0 = edge->nodes[0];
            Node* node1 = edge->nodes[1];
            Node* node2 = edge->opposites[0]->node;
            Node* node3 = edge->opposites[1]->node;
            Vector12f v(node0->v, node1->v, node2->v, node3->v);
            Vector4i indices(node0->index, node1->index, node2->index, node3->index);

            Vector12f f;
            Matrix12x12f J;
            bendingForce(edge, material, f, J);
            addMatrixAndVectorGpu(-dt * dt * J, dt * (f + dt * J * v), indices, aIndices + aIndex, aValues + aIndex, bIndices + bIndex, bValues + bIndex);
        } else
            addMatrixAndVectorGpu(Matrix12x12f(), Vector12f(), Vector4i(), aIndices + aIndex, aValues + aIndex, bIndices + bIndex, bValues + bIndex);
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

void impulseForce(const Proximity& proximity, float d, float thickness, Vector12f& f, Matrix12x12f& J) {
    float const* w = proximity.w;
    Vector3f n = proximity.n;
    float stiffness = proximity.stiffness;
    
    Vector12f N(w[0] * n, w[1] * n, w[2] * n, w[3] * n);
    f = 0.5f * stiffness / thickness * sqr(d) * N;
    J = -stiffness / thickness * d * N.outer(N);
}

void frictionForce(const Proximity& proximity, float d, float thickness, float dt, Vector12f& f, Matrix12x12f& J) {
    Node* const* nodes = proximity.nodes;
    float const* w = proximity.w;
    Vector3f n = proximity.n;
    float mu = proximity.mu;
    float stiffness = proximity.stiffness;
    
    float F = 0.5f * stiffness / thickness * sqr(d);
    Vector3f v;
    float invMass = 0.0f;
    for (int i = 0; i < 4; i++) {
        v += w[i] * nodes[i]->v;
        if (nodes[i]->isFree)
            invMass += sqr(w[i]) / nodes[i]->mass;
    }

    Matrix3x3f T = Matrix3x3f(1.0f) - n.outer(n);
    Vector3f vt = T * v;
    float f_v = min(mu * F / vt.norm(), 1.0f / (dt * invMass));
    Vector12f V(w[0] * vt, w[1] * vt, w[2] * vt, w[3] * vt);
    f = -f_v * V;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            Matrix3x3f Tij = -f_v * w[i] * w[j] * T;
            for (int k = 0; k < 3; k++)
                for (int h = 0; h < 3; h++)
                    J(3 * i + k, 3 * j + h) = Tij(k, h);
        }
}

__global__ void addProximityForcesGpu(int nProximities, const Proximity* proximities, float dt, float thickness, int nNodes, const Node* const* nodes, Pairii* aIndices, float* aValues, int* bIndices, float* bValues) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nProximities; i += nThreads) {
        const Proximity& proximity = proximities[i];
        Node* const* n = proximity.nodes;
        Vector12f v(n[0]->v, n[1]->v, n[2]->v, n[3]->v);
        Vector4i indices;
        for (int j = 0; j < 4; j++)
            indices(j) = containGpu(n[j], nNodes, nodes) ? n[j]->index : -1;

        float d = -thickness;
        for (int j = 0; j < 4; j++)
            d += proximity.w[j] * proximity.n.dot(n[j]->x);
        d = max(-d, 0.0f);


        Vector12f f;
        Matrix12x12f J;
        int aIndex, bIndex;

        impulseForce(proximity, d, thickness, f, J);
        aIndex = 288 * i;
        bIndex = 24 * i;
        addMatrixAndVectorGpu(-dt * dt * J, dt * (f + dt * J * v), indices, aIndices + aIndex, aValues + aIndex, bIndices + bIndex, bValues + bIndex);

        frictionForce(proximity, d, thickness, dt, f, J);
        aIndex = 288 * i + 144;
        bIndex = 24 * i + 12;
        addMatrixAndVectorGpu(-dt * J, dt * f, indices, aIndices + aIndex, aValues + aIndex, bIndices + bIndex, bValues + bIndex);
    }
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