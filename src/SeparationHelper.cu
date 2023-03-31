#include "SeparationHelper.cuh"

int majorAxis(const Vector3f& v) {
    return abs(v(0)) > abs(v(1)) && abs(v(0)) > abs(v(2)) ? 0 : (abs(v(1)) > abs(v(2)) ? 1 : 2);
}

bool facePlaneIntersection(const Face* face, const Face* plane, Vector3f& b0, Vector3f& b1) {
    Vector3f x0 = plane->vertices[0]->node->x;
    Vector3f n = plane->n;
    float h[3];
    int signSum = 0;
    for (int i = 0; i < 3; i++) {
        h[i] = (face->vertices[i]->node->x - x0).dot(n);
        signSum += sign(h[i]);
    }
    if (signSum == -3 || signSum == 3)
        return false;

    int v0 = -1;
    for (int i = 0; i < 3; i++)
        if (sign(h[i]) == -signSum)
            v0 = i;
    int v1 = (v0 + 1) % 3;
    int v2 = (v0 + 2) % 3;
    float t0 = h[v0] / (h[v0] - h[v1]);
    float t1 = h[v0] / (h[v0] - h[v2]);
    b0(v0) = 1.0f - t0;
    b0(v1) = t0;
    b0(v2) = 0.0f;
    b1(v0) = 1.0f - t1;
    b1(v2) = t1;
    b1(v1) = 0.0f;
    return true;
}

bool checkIntersectionMidpoint(const Face* face0, const Face* face1, Vector3f& b0, Vector3f& b1) {
    if (face0->adjacent(face1))
        return false;

    Vector3f c = face0->n.cross(face1->n);
    if (c.norm2() < 1e-12f)
        return false;

    Vector3f b00, b01, b10, b11;
    bool flag0 = facePlaneIntersection(face0, face1, b00, b01);
    bool flag1 = facePlaneIntersection(face1, face0, b10, b11);
    if (!flag0 || !flag1)
        return false;

    int axis = majorAxis(c);
    float a00 = face0->position(b00)(axis);
    float a01 = face0->position(b01)(axis);
    float a10 = face1->position(b10)(axis);
    float a11 = face1->position(b11)(axis);
    float aMin = max(min(a00, a01), min(a10, a11));
    float aMax = min(max(a00, a01), max(a10, a11));
    if (aMin > aMax)
        return false;

    float aMid = 0.5f * (aMin + aMax);
    b0 = (a00 == a01) ? b00 : b00 + (aMid - a00) / (a01 - a00) * (b01 - b00);
    b1 = (a10 == a11) ? b10 : b10 + (aMid - a10) / (a11 - a10) * (b11 - b10);
    return true;
}

__global__ void checkIntersectionsGpu(int nPairs, const PairFF* pairs, Intersection* intersections) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nPairs; i += nThreads) {
        const PairFF& pair = pairs[i];
        Face* face0 = pair.first;
        Face* face1 = pair.second;
        Intersection& intersection = intersections[i];
        if (checkIntersectionMidpoint(face0, face1, intersection.b0, intersection.b1)) {
            intersection.face0 = face0;
            intersection.face1 = face1;
        } else
            intersection.face0 = intersection.face1 = nullptr;
    }
}

__global__ void initializeOldPosition(int nIntersections, const Intersection* intersections, Vector3f* x) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nIntersections; i += nThreads) {
        const Intersection& intersection = intersections[i];

        Face* face0 = intersection.face0;
        if (!face0->isFree())
            x[2 * i] = face0->position(intersection.b0);

        Face* face1 = intersection.face1;
        if (!face1->isFree())
            x[2 * i + 1] = face1->position(intersection.b1);
    }
}

__global__ void collectContainedFaces(int nIntersections, const Intersection* intersections, int nVertices, const Vertex* const* vertices, int* indices, Vector2f* u) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nIntersections; i += nThreads) {
        const Intersection& intersection = intersections[i];

        Face* face0 = intersection.face0;
        Vector3f b0 = intersection.b0;
        int index0 = 2 * i;
        if (containGpu(face0, nVertices, vertices)) {
            indices[index0] = index0;
            u[index0] = b0(0) * face0->vertices[0]->u + b0(1) * face0->vertices[1]->u + b0(2) * face0->vertices[2]->u;
        } else
            indices[index0] = -1;

        Face* face1 = intersection.face1;
        Vector3f b1 = intersection.b1;
        int index1 = 2 * i + 1;
        if (containGpu(face1, nVertices, vertices)) {
            indices[index1] = index1;
            u[index1] = b1(0) * face1->vertices[0]->u + b1(1) * face1->vertices[1]->u + b1(2) * face1->vertices[2]->u;
        } else
            indices[index1] = -1;
    }
}

__global__ void computeEnclosingFaces(int nIndices, const Vector2f* u, int nFaces, const BackupFace* faces, int* faceIndices) {
    int nThreads = gridDim.x * blockDim.x;
    int nm = nIndices * nFaces;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nm; i += nThreads) {
        int n = i / nFaces;
        int m = i % nFaces;
        Vector3f b = faces[m].barycentricCoordinates(u[n]);
        if (b(0) >= -1e-6f && b(1) >= -1e-6f && b(2) >= -1e-5f)
            faceIndices[n] = m;
    }
}

__global__ void computeOldPositions(int nIndices, const int* indices, const int* faceIndices, const Vector2f* u, const BackupFace* faces, Vector3f* x) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nIndices; i += nThreads) {
        const BackupFace& face = faces[faceIndices[i]];
        x[indices[i]] = face.position(face.barycentricCoordinates(u[i]));
    }
}

void clearVertexFaceDistance(const Face* face0, const Face* face1, const Vector3f& d, float& maxDist, Vector3f& b0, Vector3f& b1) {
    Vector3f x0 = face1->vertices[0]->node->x;
    Vector3f x1 = face1->vertices[1]->node->x;
    Vector3f x2 = face1->vertices[2]->node->x;
    Vector3f n = face1->n;
    for (int i = 0; i < 3; i++) {
        Vector3f x = face0->vertices[i]->node->x;

        float h = (x - x0).dot(n);
        float dh = d.dot(n);
        if (h * dh >= 0.0f)
            continue;

        float a0 = mixed(x2 - x1, x - x1, d);
        float a1 = mixed(x0 - x2, x - x2, d);
        float a2 = mixed(x1 - x0, x - x0, d);
        if (a0 <= 0.0f || a1 <= 0.0f || a2 <= 0.0f)
            continue;

        float dist = -h / dh;
        if (dist > maxDist) {
            maxDist = dist;
            b0 = Vector3f();
            b0(i) = 1.0f;
            b1(0) = a0 / (a0 + a1 + a2);
            b1(1) = a1 / (a0 + a1 + a2);
            b1(2) = a2 / (a0 + a1 + a2);
        }
    }
}

void clearEdgeEdgeDistance(const Face* face0, const Face* face1, const Vector3f& d, float& maxDist, Vector3f& b0, Vector3f& b1) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            Vector3f x00 = face0->vertices[i]->node->x;
            Vector3f x01 = face0->vertices[(i + 1) % 3]->node->x;
            Vector3f x10 = face1->vertices[j]->node->x;
            Vector3f x11 = face1->vertices[(j + 1) % 3]->node->x;
            Vector3f n = (x01 - x00).normalized().cross((x11 - x10).normalized());

            float h = (x00 - x10).dot(n);
            float dh = d.dot(n);
            if (h * dh >= 0.0f)
                continue;

            float a00 = mixed(x01 - x10, x11 - x10, d);
            float a01 = mixed(x11 - x10, x00 - x10, d);
            float a10 = mixed(x01 - x00, x11 - x00, d);
            float a11 = mixed(x10 - x00, x01 - x00, d);
            if (a00 * a01 <= 0.0f || a10 * a11 <= 0.0f)
                continue;

            float dist = -h / dh;
            if (dist > maxDist) {
                maxDist = dist;
                b0 = Vector3f();
                b0(i) = a00 / (a00 + a01);
                b0((i + 1) % 3) = a01 / (a00 + a01);
                b1 = Vector3f();
                b1(j) = a10 / (a10 + a11);
                b1((j + 1) % 3) = a11 / (a10 + a11);
            }
        }
}

void farthestPoint(const Face* face0, const Face* face1, const Vector3f& d, Vector3f& b0, Vector3f& b1) {
    float maxDist = 0.0f;
    clearVertexFaceDistance(face0, face1, d, maxDist, b0, b1);
    clearVertexFaceDistance(face1, face0, -d, maxDist, b1, b0);
    clearEdgeEdgeDistance(face0, face1, d, maxDist, b0, b1);
}

__global__ void computeFarthestPoint(int nIntersections, const Vector3f* x, Intersection* intersections) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nIntersections; i += nThreads) {
        Intersection& intersection = intersections[i];
        Vector3f& d = intersection.d;
        d = (x[2 * i] - x[2 * i + 1]).normalized();
        farthestPoint(intersection.face0, intersection.face1, d, intersection.b0, intersection.b1);
    }
}