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

__global__ void collectVertexSizing(int nFaces, const Face* const* faces, const Plane* planes, int* indices, Pairfm* sizing, const Remeshing* remeshing) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads) {
        const Face* face = faces[i];
        float area = face->area;
        Matrix2x2f s = faceSizing(face, planes, remeshing);
        for (int j = 0; j < 3; j++) {
            int index = 3 * i + j;
            indices[index] = face->vertices[j]->index;
            sizing[index] = Pairfm(area, area * s);
        }
    }
}

__global__ void setVertexSizing(int nIndices, const int* indices, const Pairfm* sizing, Vertex** vertices) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nIndices; i += nThreads) {
        const Pairfm& p = sizing[i];
        Vertex* vertex = vertices[indices[i]];
        vertex->area = p.first;
        vertex->sizing = p.second / p.first;
    }
}
