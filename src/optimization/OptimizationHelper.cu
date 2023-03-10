#include "OptimizationHelper.cuh"

float clampViolation(float x, int sign) {
    return sign < 0 ? max(x, 0.0f) : (sign > 0 ? min(x, 0.0f) : x);
}

__global__ void setDiff(int nNodes, const Node* const* nodes, int* diff) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads)
        diff[i] = (i > 0 && nodes[i] != nodes[i - 1]);
}

__global__ void setIndices(int nNodes, const int* nodeIndices, const int* diff, int* indices) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads)
        indices[nodeIndices[i]] = diff[i];
}

__global__ void initializeGpu(int nNodes, const Node* const* nodes, Vector3f* x) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads)
        x[i] = nodes[i]->x;
}

__global__ void finalizeGpu(int nNodes, const Vector3f* x, Node** nodes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads)
        nodes[i]->x = x[i];
}

__global__ void collectCollisionNodes(int nConstraints, const Impact* impacts, int deform, int* indices, Node** nodes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nConstraints; i += nThreads) {
        const Impact& impact = impacts[i];
        for (int j = 0; j < 4; j++) {
            int index = 4 * i + j;
            Node* node = impact.nodes[j];
            if (deform == 1 || node->isFree) {
                indices[index] = index;
                nodes[index] = node;
            } else {
                indices[index] = -1;
                nodes[index] = nullptr;
            }
        }
    }
}

__global__ void collisionInv(int nNodes, const Node* const* nodes, float obstacleMass, float* inv) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads) {
        const Node* node = nodes[i];
        float mass = node->isFree ? node->mass : obstacleMass;
        inv[i] = 1.0f / mass;
    }
}

__global__ void collisionObjective(int nNodes, const Node* const* nodes, float obstacleMass, const Vector3f* x, float* objectives) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads) {
        const Node* node = nodes[i];
        float mass = node->isFree ? node->mass : obstacleMass;
        objectives[i] = mass * (x[i] - node->x1).norm2();
    }
}

__global__ void collisionObjectiveGradient(int nNodes, const Node* const* nodes, float invMass, float obstacleMass, const Vector3f* x, Vector3f* gradient) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads) {
        const Node* node = nodes[i];
        float mass = node->isFree ? node->mass : obstacleMass;
        gradient[i] = invMass * mass * (x[i] - node->x1);
    }
}

__global__ void collisionConstraint(int nConstraints, const Impact* impacts, const int* indices, float thickness, const Vector3f* x, float* constraints, int* signs) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nConstraints; i += nThreads) {
        float c = -thickness;
        const Impact& impact = impacts[i];
        for (int j = 0; j < 4; j++) {
            int k = indices[4 * i + j];
            if (k > -1)
                c += impact.w[j] * impact.n.dot(x[k]);
            else
                c += impact.w[j] * impact.n.dot(impact.nodes[j]->x);
        }
        constraints[i] = c;
        signs[i] = 1;
    }
}

__global__ void collectCollisionConstraintGradient(int nConstraints, const Impact* impacts, const float* coefficients, float mu, Vector3f* grad) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nConstraints; i += nThreads) {
        const Impact& impact = impacts[i];
        for (int j = 0; j < 4; j++)
            grad[4 * i + j] = mu * coefficients[i] * impact.w[j] * impact.n;
    }
}

__global__ void collectSeparationNodes(int nConstraints, const Intersection* intersections, int deform, int* indices, Node** nodes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nConstraints; i += nThreads) {
        const Intersection& intersection = intersections[i];
        Face* face0 = intersection.face0;
        Face* face1 = intersection.face1;

        for (int j = 0; j < 3; j++) {
            int index0 = 6 * i + j;
            Node* node0 = face0->vertices[j]->node;
            if (deform == 1 || node0->isFree) {
                indices[index0] = index0;
                nodes[index0] = node0;
            } else {
                indices[index0] = -1;
                nodes[index0] = nullptr;
            }

            int index1 = 6 * i + j + 3;
            Node* node1 = face1->vertices[j]->node;
            if (deform == 1 || node1->isFree) {
                indices[index1] = index1;
                nodes[index1] = node1;
            } else {
                indices[index1] = -1;
                nodes[index1] = nullptr;
            }
        }
    }
}

__global__ void separationInv(int nNodes, const Node* const* nodes, float obstacleArea, float* inv) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads) {
        const Node* node = nodes[i];
        float area = node->isFree ? node->area : obstacleArea;
        inv[i] = 1.0f / area;
    }
}

__global__ void separationObjective(int nNodes, const Node* const* nodes, float obstacleArea, const Vector3f* x, float* objectives) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads) {
        const Node* node = nodes[i];
        float area = node->isFree ? node->area : obstacleArea;
        objectives[i] = area * (x[i] - node->x1).norm2();
    }
}

__global__ void separationObjectiveGradient(int nNodes, const Node* const* nodes, float invArea, float obstacleArea, const Vector3f* x, Vector3f* gradient) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads) {
        const Node* node = nodes[i];
        float area = node->isFree ? node->area : obstacleArea;
        gradient[i] = invArea * area * (x[i] - node->x1);
    }
}

__global__ void separationConstraint(int nConstraints, const Intersection* intersections, const int* indices, float thickness, const Vector3f* x, float* constraints, int* signs) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nConstraints; i += nThreads) {
        float c = -thickness;
        const Intersection& intersection = intersections[i];
        for (int j = 0; j < 3; j++) {
            int k0 = indices[6 * i + j];
            if (k0 > -1)
                c += intersection.b0(j) * intersection.d.dot(x[k0]);
            else
                c += intersection.b0(j) * intersection.d.dot(intersection.face0->vertices[j]->node->x);

            int k1 = indices[6 * i + j + 3];
            if (k1 > -1)
                c -= intersection.b1(j) * intersection.d.dot(x[k1]);
            else
                c -= intersection.b1(j) * intersection.d.dot(intersection.face1->vertices[j]->node->x);
        }
        constraints[i] = c;
        signs[i] = 1;
    }
}

__global__ void collectSeparationConstraintGradient(int nConstraints, const Intersection* intersections, const float* coefficients, float mu, Vector3f* grad) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nConstraints; i += nThreads) {
        const Intersection& intersection = intersections[i];
        for (int j = 0; j < 3; j++) {
            grad[6 * i + j] = mu * coefficients[i] * intersection.b0(j) * intersection.d;
            grad[6 * i + j + 3] = -mu * coefficients[i] * intersection.b1(j) * intersection.d;
        }
    }
}

__global__ void addConstraintGradient(int nIndices, const int* indices, const Vector3f* grad, Vector3f* gradtient) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nIndices; i += nThreads)
        gradtient[indices[i]] += grad[i];
}

__global__ void computeCoefficient(int nConstraints, const float* lambda, float mu, const int* signs, float* c) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nConstraints; i += nThreads)
        c[i] = clampViolation(c[i] + lambda[i] / mu, signs[i]);
}

__global__ void computeSquare(int nConstraints, float* c) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nConstraints; i += nThreads)
        c[i] = sqr(c[i]);
}

__global__ void computeNorm2(int nNodes, const Vector3f* x, float* x2) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads)
        x2[i] = x[i].norm2();
}

__global__ void computeXt(int nNodes, const Vector3f* x, const Vector3f* gradient, float s, Vector3f* xt) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads)
        xt[i] = x[i] - s * gradient[i];
}

__global__ void updateX(int nNodes, const Vector3f* gradient, float s, Vector3f* x) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads)
        x[i] -= s * gradient[i];
}

__global__ void updateMultiplierGpu(int nConstraints, const float* c, const int* signs, float mu, float* lambda) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nConstraints; i += nThreads)
        lambda[i] = clampViolation(lambda[i] + mu * c[i], signs[i]);
}
