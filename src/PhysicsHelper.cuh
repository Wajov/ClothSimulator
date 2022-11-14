#ifndef PHYSICS_HELPER_CUH
#define PHYSICS_HELPER_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "MathHelper.cuh"
#include "Vector.cuh"
#include "Vertex.cuh"
#include "Edge.cuh"
#include "Face.cuh"
#include "Wind.cuh"

__global__ static void addMass(int nVertices, const Vertex* const* vertices, PairIndex* aIndices, float* aValues) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nVertices; i += nThreads) {
        float m = vertices[i]->m;
        for (int j = 0; j < 3; j++) {
            int index = 3 * i + j;
            aIndices[index] = thrust::make_pair(index, index);
            aValues[index] = m;
        }
    }
}

__global__ static void addGravity(int nVertices, const Vertex* const* vertices, float dt, const Vector3f gravity, int* bIndices, float* bValues) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nVertices; i += nThreads) {
        Vector3f g = dt * vertices[i]->m * gravity;
        for (int j = 0; j < 3; j++) {
            int index = 3 * i + j;
            bIndices[index] = index;
            bValues[index] = g(j);
        }
    }
}

__global__ static void addWindForces(int nFaces, const Face* const* faces, float dt, const Wind* wind, int* bIndices, float* bValues) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads) {
        const Face* face = faces[i];
        float area = face->getArea();
        Vector3f normal = face->getNormal();
        Vector3f average = (face->getVertex(0)->v + face->getVertex(1)->v + face->getVertex(2)->v) / 3.0f;
        Vector3f relative = wind->getVelocity() - average;
        float vn = normal.dot(relative);
        Vector3f vt = relative - vn * normal;
        Vector3f force = area * (wind->getDensity() * abs(vn) * vn * normal + wind->getDrag() * vt) / 3.0f;
        Vector3f f = dt * force;
        for (int j = 0; j < 3; j++) {
            int vertexIndex = face->getVertex(j)->index;
            for (int k = 0; k < 3; k++) {
                int index = 9 * i + 3 * j + k;
                bIndices[index] = 3 * vertexIndex + k;
                bValues[index] = f(k);
            }
        }
    }
}

__host__ __device__ static void stretchingForce(const Face* face, const Material* material, Vector9f& f, Matrix9x9f& J) {
    Matrix3x2f F = face->derivative(face->getVertex(0)->x, face->getVertex(1)->x, face->getVertex(2)->x);
    Matrix2x2f G = 0.5f * (F.transpose() * F - Matrix2x2f(1.0f));

    Matrix2x2f Y = face->getInverse();
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

    float area = face->getArea();
    f = -area * grad;
    J = -area * hess;
}

__global__ static void addStretchingForces(int nFaces, const Face* const* faces, float dt, const Material* material, PairIndex* aIndices, float* aValues, int* bIndices, float* bValues) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads) {
        const Face* face = faces[i];
        Vertex* vertex0 = face->getVertex(0);
        Vertex* vertex1 = face->getVertex(1);
        Vertex* vertex2 = face->getVertex(2);
        Vector9f v(vertex0->v, vertex1->v, vertex2->v);

        Vector9f f;
        Matrix9x9f J;
        stretchingForce(face, material, f, J);
        
        f = dt * (f + dt * J * v);
        J = -dt * dt * J;
        Vector3i indices(vertex0->index, vertex1->index, vertex2->index);
        
        for (int j = 0; j < 3; j++) {
            int x = indices(j);
            for (int k = 0; k < 3; k++) {
                int y = indices(k);
                for (int l = 0; l < 3; l++)
                    for (int r = 0; r < 3; r++) {
                        int index = 81 * i + 27 * j + 9 * k + 3 * l + r;
                        aIndices[index] = thrust::make_pair(3 * x + l, 3 * y + r);
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

__host__ __device__ static float distance(const Vector3f& x, const Vector3f& a, const Vector3f& b) {
    Vector3f e = b - a;
    Vector3f t = x - a;
    Vector3f r = e * e.dot(t) / e.norm2();
    return (t - r).norm();
}

__host__ __device__ static Vector2f barycentricWeights(const Vector3f& x, const Vector3f& a, const Vector3f& b) {
    Vector3f e = b - a;
    float t = e.dot(x - a) / e.norm2();
    return Vector2f(1.0f - t, t);
}

__host__ __device__ static void bendingForce(const Edge* edge, const Material* material, Vector12f& f, Matrix12x12f& J) {
    Vector3f x0 = edge->getVertex(0)->x;
    Vector3f x1 = edge->getVertex(1)->x;
    Vector3f x2 = edge->getOpposite(0)->x;
    Vector3f x3 = edge->getOpposite(1)->x;
    Face* adjacent0 = edge->getAdjacent(0);
    Face* adjacent1 = edge->getAdjacent(1);
    Vector3f n0 = adjacent0->getNormal();
    Vector3f n1 = adjacent1->getNormal();
    float length = edge->getLength();
    float angle = edge->getAngle();
    float area = adjacent0->getArea() + adjacent1->getArea();

    float h0 = distance(x2, x0, x1);
    float h1 = distance(x3, x0, x1);
    Vector2f w0 = barycentricWeights(x2, x0, x1);
    Vector2f w1 = barycentricWeights(x3, x0, x1);

    Vector12f dtheta(-w0(0) * n0 / h0 - w1(0) * n1 / h1, -w0(1) * n0 / h0 - w1(1) * n1 / h1, n0 / h0, n1 / h1);

    float k = material->bendingStiffness(length, angle, area, edge->getVertex(1)->u - edge->getVertex(0)->u);
    float coefficient = -0.25f * k * sqr(length) / area;

    f = coefficient * angle * dtheta;
    J = coefficient * dtheta.outer(dtheta);
}

__global__ static void addBendingForces(int nEdges, const Edge* const* edges, float dt, const Material* material, PairIndex* aIndices, float* aValues, int* bIndices, float* bValues) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads) {
        const Edge* edge = edges[i];
        Vertex* vertex0 = edge->getVertex(0);
        Vertex* vertex1 = edge->getVertex(1);
        Vertex* vertex2 = edge->getOpposite(0);
        Vertex* vertex3 = edge->getOpposite(1);

        Vector12f f;
        Matrix12x12f J;
        if (!edge->isBoundary()) {
            Vector12f v(vertex0->v, vertex1->v, vertex2->v, vertex3->v);
            
            bendingForce(edge, material, f, J);

            f = dt * (f + dt * J * v);
            J = -dt * dt * J;
        }

        Vector4i indices(vertex0->index, vertex1->index, vertex2 != nullptr ? vertex2->index : 0, vertex3 != nullptr ? vertex3->index : 0);
        
        for (int j = 0; j < 4; j++) {
            int x = indices(j);
            for (int k = 0; k < 4; k++) {
                int y = indices(k);
                for (int l = 0; l < 3; l++)
                    for (int r = 0; r < 3; r++) {
                        int index = 144 * i + 36 * j + 9 * k + 3 * l + r;
                        aIndices[index] = thrust::make_pair(3 * x + l, 3 * y + r);
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

__global__ static void addHandleForcesGpu(int nHandles, const Handle* const* handles, float dt, float stiffness, PairIndex* aIndices, float* aValues, int* bIndices, float* bValues) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nHandles; i += nThreads) {
        const Handle* handle = handles[i];
        Vertex* vertex = handle->getVertex();
        Vector3f position = handle->getPosition();
        int vertexIndex = vertex->index;
        Vector3f f = dt * ((position - vertex->x) - dt * vertex->v) * stiffness;
        for (int j = 0; j < 3; j++) {
            int index = 3 * i + j;
            aIndices[index] = thrust::make_pair(3 * vertexIndex + j, 3 * vertexIndex + j);
            aValues[index] = dt * dt * stiffness;
            bIndices[index] = 3 * vertexIndex + j;
            bValues[index] = f(j);
        }
    }
}

__global__ static void splitIndices(int nIndices, const PairIndex* indices, int* rowIndices, int* colIndices) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nIndices; i += nThreads) {
        rowIndices[i] = indices[i].first;
        colIndices[i] = indices[i].second;
    }
}

__global__ static void setupVector(int nIndices, const int* indices, const float* values, float* v) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nIndices; i += nThreads)
        v[indices[i]] = values[i];
}

__global__ static void updateVertices(int nVertices, float dt, const float* dv, Vertex** vertices) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nVertices; i += nThreads) {
        vertices[i]->x0 = vertices[i]->x;
        vertices[i]->v += Vector3f(dv[3 * i], dv[3 * i + 1], dv[3 * i + 2]);
        vertices[i]->x += vertices[i]->v * dt;
    }
}

#endif