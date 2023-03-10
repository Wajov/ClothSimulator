#ifndef BACKUP_FACE_CUH
#define BACKUP_FACE_CUH

#include <cuda_runtime.h>

#include "Vector.cuh"
#include "Matrix.cuh"

class BackupFace {
public:
    Vector3f x[3];
    Vector2f u[3];
    BackupFace();
    ~BackupFace();
    __host__ __device__ Vector3f barycentricCoordinates(const Vector2f& u) const;
    __host__ __device__ Vector3f position(const Vector3f& b) const;
};

#endif