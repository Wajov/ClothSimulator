#ifndef PROXIMITY_CUH
#define PROXIMITY_CUH

#include <cuda_runtime.h>

#include "MathHelper.cuh"
#include "Vector.cuh"
#include "Node.cuh"
#include "Edge.cuh"
#include "Face.cuh"

class Proximity {
public:
    Node* nodes[4];
    Vector3f n;
    float mu, stiffness, w[4];
    __host__ __device__ Proximity();
    __host__ __device__ Proximity(const Node* node, const Face* face, float stiffness, float clothFriction, float obstacleFriction);
    __host__ __device__ Proximity(const Edge* edge0, const Edge* edge1, float stiffness, float clothFriction, float obstacleFriction);
    __host__ __device__ ~Proximity();
};

#endif