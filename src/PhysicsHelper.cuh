#ifndef PHYSICS_HELPER_CUH
#define PHYSICS_HELPER_CHU

#include <device_launch_parameters.h>

#include "Vector.cuh"
#include "Vertex.cuh"
#include "Edge.cuh"
#include "Face.cuh"
#include "Wind.cuh"

struct VectorEntry {
    int i;
    float v;
};

struct MatrixEntry {
    int i, j;
    float v;
};

__global__ static void initGpu(int nVertices, const Vertex* const* vertices, MatrixEntry* aEntries) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nVertices; i += nThreads) {
        float m = vertices[i]->m;
        for (int j = 0; j < 3; j++) {
            int index = 3 * i + j;
            aEntries[index].i = aEntries[index].j = index;
            aEntries[index].v = m;
        }
    }
}

__global__ static void addGravityGpu(int nVertices, const Vertex* const* vertices, float dt, const Vector3f gravity, VectorEntry* bEntries) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nVertices; i += nThreads) {
        Vector3f g = dt * vertices[i]->m * gravity;
        for (int j = 0; j < 3; j++) {
            int index = 3 * i + j;
            bEntries[index].i = index;
            bEntries[index].v = g(j);
        }
    }
}

__global__ static void addWindForcesGpu(int nFaces, const Face* const* faces, float dt, const Wind* wind, VectorEntry* bEntries) {
    int nThreads = gridDim.x * blockDim.x;

    Vector3f velocity = wind->getVelocity();
    float density = wind->getDensity();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads) {
        const Face* face = faces[i];
        float area = face->getArea();
        Vector3f normal = face->getNormal();
        Vector3f average = (face->getVertex(0)->v + face->getVertex(1)->v + face->getVertex(2)->v) / 3.0f;
        Vector3f relative = velocity - average;
        float vn = normal.dot(relative);
        Vector3f vt = relative - vn * normal;
        Vector3f force = area * (density * std::abs(vn) * vn * normal + wind->getDrag() * vt) / 3.0f;
        Vector3f f = dt * force;
        for (int j = 0; j < 3; i++) {
            int vertexIndex = face->getVertex(j)->index;
            for (int k = 0; k < 3; k++) {
                int index = 9 * i + 3 * j + k;
                bEntries[index].i = 3 * vertexIndex + k;
                bEntries[index].v = f(k);
            }
        }
    }
}

__global__ static void addStretchingForcesGpu() {
    // TODO
}

__global__ static void addBendingForcesGpu() {
    // TODO
}

#endif