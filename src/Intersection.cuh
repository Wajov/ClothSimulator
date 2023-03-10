#ifndef INTERSECTION_CUH
#define INTERSECTION_CUH

#include <cuda_runtime.h>

#include "Vector.cuh"
#include "Face.cuh"

class Intersection {
public:
    Face* face0, * face1;
    Vector3f b0, b1, d;
    __host__ __device__ Intersection();
    __host__ __device__ ~Intersection();
};

#endif