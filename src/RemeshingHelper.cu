#include "RemeshingHelper.cuh"

__global__ void setX(int nNodes, const Node* const* nodes, Vector3f* x) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads)
        x[i] = nodes[i]->x;
}

__global__ void initializeNearPoints(int nNodes, const Vector3f* x, NearPoint* points) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads)
        points[i].x = x[i];
}

float unsignedVertexEdgeDistance(const Vector3f& x, const Vector3f& y0, const Vector3f& y1, Vector3f& n, float& wx, float& wy0, float& wy1) {
    float t = clamp((x - y0).dot(y1 - y0)/(y1 - y0).dot(y1 - y0), 0.0f, 1.0f);
    Vector3f y = y0 + t * (y1 - y0);
    float d = (x - y).norm();
    n = (x - y).normalized();
    wx = 1.0f;
    wy0 = 1.0f - t;
    wy1 = t;
    return d;
}

float unsignedVertexFaceDistance(const Vector3f& x, const Vector3f& y0, const Vector3f& y1, const Vector3f& y2, Vector3f& n, float* w) {
    Vector3f nt = (y1 - y0).cross(y2 - y0).normalized();
    float d = abs((x - y0).dot(nt));
    float b0 = mixed(y1 - x, y2 - x, nt);
    float b1 = mixed(y2 - x, y0 - x, nt);
    float b2 = mixed(y0 - x, y1 - x, nt);
    if (b0 >= 0.0f && b1 >= 0.0f && b2 >= 0.0f) {
        n = nt;
        w[0] = 1.0f;
        w[1] = -b0 / (b0 + b1 + b2);
        w[2] = -b1 / (b0 + b1 + b2);
        w[3] = -b2 / (b0 + b1 + b2);
        return d;
    }
    d = INFINITY;
    if (b0 < 0.0f) {
        float dt = unsignedVertexEdgeDistance(x, y1, y2, n, w[0], w[2], w[3]);
        if (dt < d) {
            d = dt;
            w[1] = 0.0f;
        }
    }
    if (b1 < 0.0f) {
        float dt = unsignedVertexEdgeDistance(x, y2, y0, n, w[0], w[3], w[1]);
        if (dt < d) {
            d = dt;
            w[2] = 0.0f;
        }
    }
    if (b2 < 0.0f) {
        float dt = unsignedVertexEdgeDistance(x, y0, y1, n, w[0], w[1], w[2]);
        if (dt < d) {
            d = dt;
            w[3] = 0.0f;
        }
    }
    return d;
}

void checkNearestPoint(const Vector3f& x, const Face* face, NearPoint& point) {
    Vector3f n;
    float w[4];
    Vector3f x1 = face->vertices[0]->node->x;
    Vector3f x2 = face->vertices[1]->node->x;
    Vector3f x3 = face->vertices[2]->node->x;
    float d = unsignedVertexFaceDistance(x, x1, x2, x3, n, w);

    if (d < point.d) {
        point.d = d;
        point.x = -(w[1] * x1 + w[2] * x2 + w[3] * x3);
    }
}

__global__ void setNearestPlane(int nNodes, const Vector3f* x, const NearPoint* points, Plane* planes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads) {
        Vector3f xt = points[i].x;
        Vector3f n = x[i] - xt;
        if (n.norm2() > 1e-8f)
            planes[i] = Plane(xt, n.normalized());
    }
}

Matrix2x2f diagonal(const Vector2f& v) {
    Matrix2x2f ans;
    for (int i = 0; i < 2; i++)
        ans(i, i) = v(i);
    return ans;
}

Matrix2x2f sqrt(const Matrix2x2f& A) {
    Matrix2x2f Q;
    Vector2f l;
    eigenvalueDecomposition(A, Q, l);
    for (int i = 0; i < 2; i++)
        l(i) = l(i) >= 0.0f ? sqrt(l(i)) : -sqrt(-l(i));
    return Q * diagonal(l) * Q.transpose();
}

Matrix2x2f max(const Matrix2x2f& A, float v) {
    Matrix2x2f Q;
    Vector2f l;
    eigenvalueDecomposition(A, Q, l);
    for (int i = 0; i < 2; i++)
        l(i) = max(l(i), v);
    return Q * diagonal(l) * Q.transpose();
}

Matrix2x2f compressionMetric(const Matrix2x2f& G, const Matrix2x2f& S2, const Remeshing* remeshing) {
    Matrix2x2f P(Vector2f(S2(1, 1), -S2(1, 0)), Vector2f(-S2(0, 1), S2(0, 0)));
    Matrix2x2f D = G.transpose() * G - 4.0f * sqr(remeshing->refineCompression) * P * remeshing->ribStiffening;
    return max(-G + sqrt(D), 0.0f) / (2.0f * sqr(remeshing->refineCompression));
}

Matrix2x2f obstacleMetric(const Face* face, const Plane* planes) {
    Matrix2x2f ans;
    for (int i = 0; i < 3; i++) {
        Plane plane = planes[face->vertices[i]->node->index];
        if (plane.n.norm2() == 0.0f)
            continue;
        float h[3];
        for (int j = 0; j < 3; j++)
            h[j] = (face->vertices[j]->node->x - plane.p).dot(plane.n);
        Vector2f dh = face->inverse.transpose() * Vector2f(h[1] - h[0], h[2] - h[0]);
        ans += dh.outer(dh) / sqr(h[i]);
    }
    return ans / 3.0f;
}

Disk circumscribedDisk(const Disk& d0, const Disk& d1) {
    float d = (d0.o - d1.o).norm();
    float r = 0.5f * (d0.r + d + d1.r);
    float t = (r - d0.r) / d;
    return Disk(d0.o + t * (d1.o - d0.o), r);
}

Disk circumscribedDisk(const Disk& d0, const Disk& d1, const Disk& d2) {
    float x0 = d0.o(0), y0 = d0.o(1), r0 = d0.r;
    float x1 = d1.o(0), y1 = d1.o(1), r1 = d1.r;
    float x2 = d2.o(0), y2 = d2.o(1), r2 = d2.r;

    float v11 = 2.0f * x1 - 2.0f * x0;
    float v12 = 2.0f * y1 - 2.0f * y0;
    float v13 = sqr(x0) - sqr(x1) + sqr(y0) - sqr(y1) - sqr(r0) + sqr(r1);
    float v14 = 2.0f * r1 - 2.0f * r0;
    float v21 = 2.0f * x2 - 2.0f * x1;
    float v22 = 2.0f * y2 - 2.0f * y1;
    float v23 = sqr(x1) - sqr(x2) + sqr(y1) - sqr(y2) - sqr(r1) + sqr(r2);
    float v24 = 2.0f * r2 - 2.0f * r1;
    float w12 = v12 / v11;
    float w13 = v13 / v11;
    float w14 = v14 / v11;
    float w22 = v22 / v21 - w12;
    float w23 = v23 / v21 - w13;
    float w24 = v24 / v21 - w14;
    float P = -w23 / w22;
    float Q = w24 / w22;
    float M = - w12 * P - w13;
    float N = w14 - w12 * Q;
    float a = sqr(N) + sqr(Q) - 1.0f;
    float b = 2.0f * M * N - 2.0f * N * x0 + 2.0f * P * Q - 2.0f * Q * y0 + 2.0f * r0;
    float c = sqr(x0) + sqr(M) - 2.0f * M * x0 + sqr(P) + sqr(y0) - 2.0f * P * y0 - sqr(r0);
    float D = sqr(b) - 4.0f * a * c;
    float rs = (-b - sqrt(D)) / (2.0f * a);
    float xs = M + N * rs;
    float ys = P + Q * rs;

    return Disk(Vector2f(xs , ys), rs);
}

Matrix2x2f maxTensor(const Matrix2x2f* M) {
    int n = 0;
    Disk d[5];
    for (int i = 0; i < 5; i++)
        if (M[i].trace() != 0.0f) {
            d[n].o = Vector2f(0.5f * (M[i](0, 0) - M[i](1, 1)), 0.5f * (M[i](0, 1) + M[i](1, 0)));
            d[n].r = 0.5f * (M[i](0, 0) + M[i](1, 1));
            n++;
        }

    Disk disk;
    disk = d[0];
    for (int i = 1; i < n; i++)
        if (!disk.enclose(d[i])) {
            disk = d[i];
            for (int j = 0; j < i; j++)
                if (!disk.enclose(d[j])) {
                    disk = circumscribedDisk(d[i], d[j]);
                    for (int k = 0; k < j; k++)
                        if (!disk.enclose(d[k]))
                            disk = circumscribedDisk(d[i], d[j], d[k]);
                }
        }

    Matrix2x2f ans;
    ans(0, 0) = disk.r + disk.o(0);
    ans(0, 1) = ans(1, 0) = disk.o(1);
    ans(1, 1) = disk.r - disk.o(0);
    return ans;
}

Matrix2x2f faceSizing(const Face* face, const Plane* planes, const Remeshing* remeshing) {
    Node* node0 = face->vertices[0]->node;
    Node* node1 = face->vertices[1]->node;
    Node* node2 = face->vertices[2]->node;
    Matrix2x2f M[5];

    Matrix2x2f Sw1 = face->curvature();
    M[0] = (Sw1.transpose() * Sw1) / sqr(remeshing->refineAngle);
    Matrix3x2f Sw2 = face->derivative(node0->n, node1->n, node2->n);
    M[1] = (Sw2.transpose() * Sw2) / sqr(remeshing->refineAngle);
    Matrix3x2f V = face->derivative(node0->v, node1->v, node2->v);
    M[2] = (V.transpose() * V) / sqr(remeshing->refineVelocity);
    Matrix3x2f F = face->derivative(node0->x, node1->x, node2->x);
    M[3] = compressionMetric(F.transpose() * F - Matrix2x2f(1.0f), Sw2.transpose() * Sw2, remeshing);
    M[4] = obstacleMetric(face, planes);
    Matrix2x2f S = maxTensor(M);

    Matrix2x2f Q;
    Vector2f l;
    eigenvalueDecomposition(S, Q, l);
    for (int i = 0; i < 2; i++)
        l(i) = clamp(l(i), 1.0f / sqr(remeshing->sizeMax), 1.0f / sqr(remeshing->sizeMin));
    float lMax = max(l(0), l(1));
    float lMin = lMax * sqr(remeshing->aspectMin);
    for (int i = 0; i < 2; i++)
        l(i) = max(l(i), lMin);
    return Q * diagonal(l) * Q.transpose();
}

__global__ void initializeSizing(int nVertices, Vertex** vertices) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nVertices; i += nThreads) {
        Vertex* vertex = vertices[i];
        vertex->area = 0.0f;
        vertex->sizing = Matrix2x2f();
    }
}

__global__ void computeSizingGpu(int nFaces, const Face* const* faces, const Plane* planes, const Remeshing* remeshing) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads) {
        const Face* face = faces[i];
        float area = face->area;
        Matrix2x2f sizing = faceSizing(face, planes, remeshing);
        for (int j = 0; j < 3; j++) {
            Vertex* vertex = face->vertices[j];
            atomicAdd(&vertex->area, area);
            for (int k = 0; k < 2; k++)
                for (int h = 0; h < 2; h++)
                    atomicAdd(&vertex->sizing(k, h), area * sizing(k, h));
        }
    }
}

__global__ void finalizeSizing(int nVertices, Vertex** vertices) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nVertices; i += nThreads) {
        Vertex* vertex = vertices[i];
        vertex->sizing /= vertex->area;
    }
}

bool shouldFlip(const Edge* edge, const Remeshing* remeshing) {
    if (edge->isBoundary() || edge->isSeam())
        return false;
        
    Vertex* vertex0 = edge->vertices[0][0];
    Vertex* vertex1 = edge->vertices[1][1];
    Vertex* vertex2 = edge->opposites[0];
    Vertex* vertex3 = edge->opposites[1];

    Vector2f x = vertex0->u, y = vertex1->u, z = vertex2->u, w = vertex3->u;
    Matrix2x2f M = 0.25f * (vertex0->sizing + vertex1->sizing + vertex2->sizing + vertex3->sizing);
    float area0 = edge->adjacents[0]->area;
    float area1 = edge->adjacents[1]->area;
    return area1 * (x - z).dot(M * (y - z)) + area0 * (y - w).dot(M * (x - w)) < -remeshing->flipThreshold * (area0 + area1);
}

__global__ void checkEdgesToFlip(int nEdges, const Edge* const* edges, const Remeshing* remeshing, Edge** edgesToFlip) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads) {
        const Edge* edge = edges[i];
        edgesToFlip[i] = shouldFlip(edge, remeshing) ? const_cast<Edge*>(edge) : nullptr;
    }
}

__global__ void initializeEdgeNodes(int nEdges, const Edge* const* edges) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads) {
        const Edge* edge = edges[i];
        for (int j = 0; j < 2; j++)
            edge->nodes[j]->removed = false;
    }
}

__global__ void resetEdgeNodes(int nEdges, const Edge* const* edges) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads) {
        const Edge* edge = edges[i];
        Node* node0 = edge->nodes[0];
        Node* node1 = edge->nodes[1];
        if (!node0->removed && !node1->removed)
            node0->minIndex = node1->minIndex = nEdges;
    }
}

__global__ void computeEdgeMinIndices(int nEdges, const Edge* const* edges) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads) {
        const Edge* edge = edges[i];
        Node* node0 = edge->nodes[0];
        Node* node1 = edge->nodes[1];
        if (!node0->removed && !node1->removed) {
            atomicMin(&node0->minIndex, i);
            atomicMin(&node1->minIndex, i);
        }
    }
}

__global__ void checkIndependentEdges(int nEdges, const Edge* const* edges, Edge** independentEdges) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads) {
        const Edge* edge = edges[i];
        Node* node0 = edge->nodes[0];
        Node* node1 = edge->nodes[1];
        if (!node0->removed && !node1->removed && node0->minIndex == node1->minIndex) {
            independentEdges[i] = const_cast<Edge*>(edge);
            node0->removed = node1->removed = true;
        }
    }
}

__global__ void flipGpu(int nEdges, const Edge* const* edges, const Material* material, Edge** addedEdges, Edge** removedEdges, Face** addedFaces, Face** removedFaces) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads) {
        const Edge* edge = edges[i];
        Vertex* vertex0 = edge->vertices[0][0];
        Vertex* vertex1 = edge->vertices[1][1];
        Vertex* vertex2 = edge->opposites[0];
        Vertex* vertex3 = edge->opposites[1];

        Face* face0 = edge->adjacents[0];
        Face* face1 = edge->adjacents[1];

        Edge* edge0 = face0->findEdge(vertex1, vertex2);
        Edge* edge1 = face0->findEdge(vertex2, vertex0);
        Edge* edge2 = face1->findEdge(vertex0, vertex3);
        Edge* edge3 = face1->findEdge(vertex3, vertex1);

        Edge* newEdge = new Edge(vertex2->node, vertex3->node);
        Face* newFace0 = new Face(vertex0, vertex3, vertex2, material);
        Face* newFace1 = new Face(vertex1, vertex2, vertex3, material);
        newEdge->initialize(vertex0, newFace0);
        newEdge->initialize(vertex1, newFace1);
        newFace0->setEdges(edge2, newEdge, edge1);
        newFace1->setEdges(edge0, newEdge, edge3);

        edge0->initialize(vertex3, newFace1);
        edge1->initialize(vertex3, newFace0);
        edge2->initialize(vertex2, newFace0);
        edge3->initialize(vertex2, newFace1);

        addedEdges[i] = newEdge;
        removedEdges[i] = const_cast<Edge*>(edge);
        addedFaces[2 * i] = newFace0;
        addedFaces[2 * i + 1] = newFace1;
        removedFaces[2 * i] = face0;
        removedFaces[2 * i + 1] = face1;
    }
}

float edgeMetric(const Vertex* vertex0, const Vertex* vertex1) {
    if (vertex0 == nullptr || vertex1 == nullptr)
        return 0.0f;
    Vector2f du = vertex0->u - vertex1->u;
    return sqrt(0.5f * (du.dot(vertex0->sizing * du) + du.dot(vertex1->sizing * du)));
}

float edgeMetric(const Edge* edge) {
    return max(edgeMetric(edge->vertices[0][0], edge->vertices[0][1]), edgeMetric(edge->vertices[1][0], edge->vertices[1][1]));
}

__global__ void checkEdgesToSplit(int nEdges, const Edge* const* edges, Edge** edgesToSplit, float* metrics) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads) {
        const Edge* edge = edges[i];
        float m = edgeMetric(edge);
        if (m > 1.0f) {
            edgesToSplit[i] = const_cast<Edge*>(edge);
            metrics[i] = m;
        } else
            edgesToSplit[i] = nullptr;
    }
}

__global__ void splitGpu(int nEdges, const Edge* const* edges, const Material* material, Node** addedNodes, Vertex** addedVertices, Edge** addedEdges, Edge** removedEdges, Face** addedFaces, Face** removedFaces) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads) {
        const Edge* edge = edges[i];
        Node* node0 = edge->nodes[0];
        Node* node1 = edge->nodes[1];

        Node* newNode = new Node(0.5f * (node0->x + node1->x), node0->isFree && node1->isFree);
        newNode->x0 = 0.5f * (node0->x0 + node1->x0);
        newNode->v = 0.5f * (node0->v + node1->v);
        Edge* newEdges[2];
        newEdges[0] = new Edge(newNode, node0);
        newEdges[1] = new Edge(newNode, node1);
        
        addedNodes[i] = newNode;
        addedEdges[4 * i] = newEdges[0];
        addedEdges[4 * i + 1] = newEdges[1];
        removedEdges[i] = const_cast<Edge*>(edge);

        Vertex* newVertices[2];
        if (edge->isSeam()) {
            newVertices[0] = new Vertex(0.5f * (edge->vertices[0][0]->u + edge->vertices[0][1]->u));
            newVertices[0]->sizing = 0.5f * (edge->vertices[0][0]->sizing + edge->vertices[0][1]->sizing);
            newVertices[1] = new Vertex(0.5f * (edge->vertices[1][0]->u + edge->vertices[1][1]->u));
            newVertices[1]->sizing = 0.5f * (edge->vertices[1][0]->sizing + edge->vertices[1][1]->sizing);
            newVertices[0]->node = newVertices[1]->node = newNode;
            addedVertices[2 * i] = newVertices[0];
            addedVertices[2 * i + 1] = newVertices[1];
        } else {
            int j = edge->opposites[0] != nullptr ? 0 : 1;
            newVertices[0] = newVertices[1] = new Vertex(0.5f * (edge->vertices[j][0]->u + edge->vertices[j][1]->u));
            newVertices[0]->sizing = 0.5f * (edge->vertices[j][0]->sizing + edge->vertices[j][1]->sizing);
            newVertices[0]->node = newNode;
            addedVertices[2 * i] = newVertices[0];
            addedVertices[2 * i + 1] = nullptr;
        }

        for (int j = 0; j < 2; j++)
            if (edge->opposites[j] != nullptr) {
                Vertex* vertex0 = edge->vertices[j][j];
                Vertex* vertex1 = edge->vertices[j][1 - j];
                Vertex* vertex2 = edge->opposites[j];

                Face* face = edge->adjacents[j];
                Edge* edge0 = face->findEdge(vertex1, vertex2);
                Edge* edge1 = face->findEdge(vertex2, vertex0);
                
                Vertex* newVertex = newVertices[j];
                Edge* newEdge0 = newEdges[j];
                Edge* newEdge1 = newEdges[1 - j];
                Edge* newEdge2 = new Edge(newNode, vertex2->node);
                Face* newFace0 = new Face(vertex0, newVertex, vertex2, material);
                Face* newFace1 = new Face(vertex2, newVertex, vertex1, material);
                
                newEdge0->initialize(vertex2, newFace0);
                newEdge1->initialize(vertex2, newFace1);
                newEdge2->initialize(vertex0, newFace0);
                newEdge2->initialize(vertex1, newFace1);
                newFace0->setEdges(newEdge0, newEdge2, edge1);
                newFace1->setEdges(newEdge2, newEdge1, edge0);
                edge0->initialize(newVertex, newFace1);
                edge1->initialize(newVertex, newFace0);
                
                addedEdges[4 * i + j + 2] = newEdge2;
                addedFaces[4 * i + 2 * j] = newFace0;
                addedFaces[4 * i + 2 * j + 1] = newFace1;
                removedFaces[2 * i + j] = face;
            } else {
                addedEdges[4 * i + j + 2] = nullptr;
                addedFaces[4 * i + 2 * j] = addedFaces[4 * i + 2 * j + 1] = nullptr;
                removedFaces[2 * i + j] = nullptr;
            }
    }
}

__global__ void collectAdjacentEdges(int nEdges, const Edge* const* edges, int* indices, Edge** adjacentEdges) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads) {
        Edge* edge = const_cast<Edge*>(edges[i]);
        for (int j = 0; j < 2; j++) {
            int index = 2 * i + j;
            indices[index] = edge->nodes[j]->index;
            adjacentEdges[index] = edge;
        }
    }
}

__global__ void collectAdjacentFaces(int nFaces, const Face* const* faces, int* indices, Face** adjacentFaces) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads) {
        Face* face = const_cast<Face*>(faces[i]);
        for (int j = 0; j < 3; j++) {
            int index = 3 * i + j;
            indices[index] = face->vertices[j]->index;
            adjacentFaces[index] = face;
        }
    }
}

__global__ void setRange(int n, const int* indices, int* l, int* r) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += nThreads) {
        int index = indices[i];
        if (i == 0 || index != indices[i - 1])
            l[index] = i;
        if (i == n - 1 || index != indices[i + 1])
            r[index] = i + 1;
    }
}

__device__ bool shouldCollapseGpu(const Edge* edge, int side, const int* edgeBegin, const int* edgeEnd, const Edge* const* adjacentEdges, const int* faceBegin, const int* faceEnd, const Face* const* adjacentFaces, const Remeshing* remeshing) {
    Node* node = edge->nodes[side];
    if (node->preserve)
        return false;
    
    bool flag = false;
    int l = edgeBegin[node->index], r = edgeEnd[node->index];
    for (int i = l; i < r; i++) {
        const Edge* adjacentEdge = adjacentEdges[i];
        if (adjacentEdge->isBoundary() || adjacentEdge->isSeam()) {
            flag = true;
            break;
        }
    }
    if (flag && (!edge->isBoundary() && !edge->isSeam()))
        return false;
    
    if (edge->isSeam())
        for (int i = 0; i < 2; i++) {
            Vertex* vertex0 = edge->vertices[i][side];
            Vertex* vertex1 = edge->vertices[i][1 - side];
            
            int l = faceBegin[vertex0->index], r = faceEnd[vertex0->index];
            for (int j = l; j < r; j++) {
                const Face* adjacentFace = adjacentFaces[j];
                Vertex* v0 = adjacentFace->vertices[0];
                Vertex* v1 = adjacentFace->vertices[1];
                Vertex* v2 = adjacentFace->vertices[2];
                if (v0 == vertex1 || v1 == vertex1 || v2 == vertex1)
                    continue;
                
                if (v0 == vertex0)
                    v0 = vertex1;
                else if (v1 == vertex0) {
                    v1 = vertex1;
                    mySwap(v0, v1);
                } else {
                    v2 = vertex1;
                    mySwap(v0, v2);
                }
                Vector2f u0 = v0->u;
                Vector2f u1 = v1->u;
                Vector2f u2 = v2->u;
                float a = 0.5f * (u1 - u0).cross(u2 - u0);
                float p = (u0 - u1).norm() + (u1 - u2).norm() + (u2 - u0).norm();
                float aspect = 12.0f * sqrt(3.0f) * a / sqr(p);
                if (a < 1e-6f || aspect < remeshing->aspectMin)
                    return false;
                if (edgeMetric(v0, v1) > 0.9f || edgeMetric(v0, v2) > 0.9f)
                    return false;
            }
        }
    else {
        int index = edge->opposites[0] != nullptr ? 0 : 1;
        Vertex* vertex0 = edge->vertices[index][side];
        Vertex* vertex1 = edge->vertices[index][1 - side];

        int l = faceBegin[vertex0->index], r = faceEnd[vertex0->index];
        for (int i = l; i < r; i++) {
            const Face* adjacentFace = adjacentFaces[i];
            Vertex* v0 = adjacentFace->vertices[0];
            Vertex* v1 = adjacentFace->vertices[1];
            Vertex* v2 = adjacentFace->vertices[2];
            if (v0 == vertex1 || v1 == vertex1 || v2 == vertex1)
                continue;
            
            if (v0 == vertex0)
                v0 = vertex1;
            else if (v1 == vertex0) {
                v1 = vertex1;
                mySwap(v0, v1);
            } else {
                v2 = vertex1;
                mySwap(v0, v2);
            }
            Vector2f u0 = v0->u;
            Vector2f u1 = v1->u;
            Vector2f u2 = v2->u;
            float a = 0.5f * (u1 - u0).cross(u2 - u0);
            float p = (u0 - u1).norm() + (u1 - u2).norm() + (u2 - u0).norm();
            float aspect = 12.0f * sqrt(3.0f) * a / sqr(p);
            if (a < 1e-6f || aspect < remeshing->aspectMin)
                return false;
            if (edgeMetric(v0, v1) > 0.9f || edgeMetric(v0, v2) > 0.9f)
                return false;
        }
    }

    return true;
}

__global__ void checkEdgesToCollapse(int nEdges, const Edge* const* edges, const int* edgeBegin, const int* edgeEnd, const Edge* const* adjacentEdges, const int* faceBegin, const int* faceEnd, const Face* const* adjacentFaces, const Remeshing* remeshing, Pairei* edgesToCollapse) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads) {
        const Edge* edge = edges[i];
        if (shouldCollapseGpu(edge, 0, edgeBegin, edgeEnd, adjacentEdges, faceBegin, faceEnd, adjacentFaces, remeshing))
            edgesToCollapse[i] = Pairei(const_cast<Edge*>(edge), 0);
        else if (shouldCollapseGpu(edge, 1, edgeBegin, edgeEnd, adjacentEdges, faceBegin, faceEnd, adjacentFaces, remeshing))
            edgesToCollapse[i] = Pairei(const_cast<Edge*>(edge), 1);
        else
            edgesToCollapse[i] = Pairei(nullptr, -1);
    }
}

__global__ void initializeCollapseNodes(int nEdges, const Pairei* edges, const int* edgeBegin, const int* edgeEnd, const Edge* const* adjacentEdges) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads) {
        Edge* edge = edges[i].first;
        int side = edges[i].second;
        
        Node* node = edge->nodes[side];
        node->removed = false;
        int l = edgeBegin[node->index], r = edgeEnd[node->index];
        for (int j = l; j < r; j++) {
            const Edge* adjacentEdge = adjacentEdges[j];
            Node* adjacentNode = adjacentEdge->nodes[0] != node ? adjacentEdge->nodes[0] : adjacentEdge->nodes[1];
            adjacentNode->removed = false;
        }
    }
}

__global__ void resetCollapseNodes(int nEdges, const Pairei* edges, const int* edgeBegin, const int* edgeEnd, const Edge* const* adjacentEdges) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads) {
        Edge* edge = edges[i].first;
        int side = edges[i].second;

        Node* node = edge->nodes[side];
        if (!node->removed) {
            int l = edgeBegin[node->index], r = edgeEnd[node->index];
            bool flag = true;
            for (int j = l; j < r; j++) {
                const Edge* adjacentEdge = adjacentEdges[j];
                Node* adjacentNode = adjacentEdge->nodes[0] != node ? adjacentEdge->nodes[0] : adjacentEdge->nodes[1];
                if (adjacentNode->removed) {
                    flag = false;
                    break;
                }
            }

            if (flag) {
                node->minIndex = nEdges;
                for (int j = l; j < r; j++) {
                    const Edge* adjacentEdge = adjacentEdges[j];
                    Node* adjacentNode = adjacentEdge->nodes[0] != node ? adjacentEdge->nodes[0] : adjacentEdge->nodes[1];
                    adjacentNode->minIndex = nEdges;
                }
            }
        }
    }
}

__global__ void computeCollapseMinIndices(int nEdges, const Pairei* edges, const int* edgeBegin, const int* edgeEnd, const Edge* const* adjacentEdges) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads) {
        Edge* edge = edges[i].first;
        int side = edges[i].second;

        Node* node = edge->nodes[side];
        if (!node->removed) {
            int l = edgeBegin[node->index], r = edgeEnd[node->index];
            bool flag = true;
            for (int j = l; j < r; j++) {
                const Edge* adjacentEdge = adjacentEdges[j];
                Node* adjacentNode = adjacentEdge->nodes[0] != node ? adjacentEdge->nodes[0] : adjacentEdge->nodes[1];
                if (adjacentNode->removed) {
                    flag = false;
                    break;
                }
            }

            if (flag) {
                atomicMin(&node->minIndex, i);
                for (int j = l; j < r; j++) {
                    const Edge* adjacentEdge = adjacentEdges[j];
                    Node* adjacentNode = adjacentEdge->nodes[0] != node ? adjacentEdge->nodes[0] : adjacentEdge->nodes[1];
                    atomicMin(&adjacentNode->minIndex, i);
                }
            }
        }
    }
}

__global__ void checkIndependentEdgesToCollapse(int nEdges, const Pairei* edges, const int* edgeBegin, const int* edgeEnd, const Edge* const* adjacentEdges, Pairei* edgesToCollapse) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads) {
        Edge* edge = edges[i].first;
        int side = edges[i].second;

        Node* node = edge->nodes[side];
        if (!node->removed && node->minIndex == i) {
            int l = edgeBegin[node->index], r = edgeEnd[node->index];
            bool flag = true;
            for (int j = l; j < r; j++) {
                const Edge* adjacentEdge = adjacentEdges[j];
                Node* adjacentNode = adjacentEdge->nodes[0] != node ? adjacentEdge->nodes[0] : adjacentEdge->nodes[1];
                if (adjacentNode->removed || adjacentNode->minIndex != i) {
                    flag = false;
                    break;
                }
            }

            if (flag) {
                edgesToCollapse[i] = edges[i];
                node->removed = true;
                for (int j = l; j < r; j++) {
                    const Edge* adjacentEdge = adjacentEdges[j];
                    Node* adjacentNode = adjacentEdge->nodes[0] != node ? adjacentEdge->nodes[0] : adjacentEdge->nodes[1];
                    adjacentNode->removed = true;
                }
            }
        }
    }
}

__global__ void collapseGpu(int nEdges, const Pairei* edges, const Material* material, const int* edgeBegin, const int* edgeEnd, Edge* const* adjacentEdges, const int* faceBegin, const int* faceEnd, Face* const* adjacentFaces, Node** removedNodes, Vertex** removedVertices, Edge** removedEdges, Face** removedFaces) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads) {
        Edge* edge = edges[i].first;
        int side = edges[i].second;
        Node* node0 = edge->nodes[side];
        Node* node1 = edge->nodes[1 - side];

        removedNodes[i] = node0;
        removedEdges[3 * i] = edge;

        for (int j = 0; j < 2; j++)
            if (edge->opposites[j] != nullptr) {
                Vertex* vertex0 = edge->vertices[j][side];
                Vertex* vertex1 = edge->vertices[j][1 - side];
                Vertex* vertex2 = edge->opposites[j];

                Face* face = edge->adjacents[j];
                Edge* edge0 = face->findEdge(vertex1, vertex2);
                Edge* edge1 = face->findEdge(vertex2, vertex0);

                if (!edge1->isBoundary()) {
                    Vertex* v = edge1->opposites[0];
                    Face* f = edge1->adjacents[0];
                    if (f == face) {
                        v = edge1->opposites[1];
                        f = edge1->adjacents[1];
                    }

                    edge0->replaceOpposite(vertex0, v);
                    edge0->replaceAdjacent(face, f);
                    f->replaceEdge(edge1, edge0);
                }

                removedEdges[3 * i + j + 1] = edge1;
                removedFaces[2 * i + j] = face;
            } else {
                removedEdges[3 * i + j + 1] = nullptr;
                removedFaces[2 * i + j] = nullptr;
            }

        int l = edgeBegin[node0->index], r = edgeEnd[node0->index];
        for (int j = l; j < r; j++) {
            Edge* adjacentEdge = adjacentEdges[j];
            if (adjacentEdge != edge) {
                adjacentEdge->replaceNode(node0, node1);
                adjacentEdge->replaceVertex(edge->vertices[0][side], edge->vertices[0][1 - side]);
                adjacentEdge->replaceVertex(edge->vertices[1][side], edge->vertices[1][1 - side]);
            }
        }

        if (edge->isSeam())
            for (int j = 0; j < 2; j++) {
                Vertex* vertex0 = edge->vertices[j][side];
                Vertex* vertex1 = edge->vertices[j][1 - side];
                removedVertices[2 * i + j] = vertex0;

                l = faceBegin[vertex0->index];
                r = faceEnd[vertex0->index];
                for (int k = l; k < r; k++) {
                    Face* adjacentFace = adjacentFaces[k];
                    if (adjacentFace != edge->adjacents[0] && adjacentFace != edge->adjacents[1]) {
                        adjacentFace->findOpposite(vertex0)->replaceOpposite(vertex0, vertex1);
                        adjacentFace->replaceVertex(vertex0, vertex1);
                        adjacentFace->initialize(material);
                    }
                }
            }
        else {
            int index = edge->opposites[0] != nullptr ? 0 : 1;
            Vertex* vertex0 = edge->vertices[index][side];
            Vertex* vertex1 = edge->vertices[index][1 - side];
            removedVertices[2 * i] = vertex0;
            removedVertices[2 * i + 1] = nullptr;
            
            l = faceBegin[vertex0->index];
            r = faceEnd[vertex0->index];
            for (int k = l; k < r; k++) {
                Face* adjacentFace = adjacentFaces[k];
                if (adjacentFace != edge->adjacents[0] && adjacentFace != edge->adjacents[1]) {
                    adjacentFace->findOpposite(vertex0)->replaceOpposite(vertex0, vertex1);
                    adjacentFace->replaceVertex(vertex0, vertex1);
                    adjacentFace->initialize(material);
                }
            }
        }
    }
}

__global__ void printEdges(int nEdges, const Pairei* edges) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads) {
        Edge* edge = edges[i].first;
        int side = edges[i].second;
        printf("%d %d %d\n", edge->nodes[0]->index, edge->nodes[1]->index, side);
    }
}