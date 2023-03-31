#ifndef CUDA_HELPER_CUH
#define CUDA_HELPER_CUH

#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <thrust/device_vector.h>

#include "Pair.cuh"
#include "Impact.cuh"
#include "Intersection.cuh"
#include "Proximity.cuh"

#define CUDA_CHECK(val) cudaCheck((val), #val, __FILE__, __LINE__)
#define CUDA_CHECK_LAST() cudaCheckLast(__FILE__, __LINE__)
#define CUBLAS_CHECK(val) cublasCheck((val), #val, __FILE__, __LINE__)
#define CUSPARSE_CHECK(val) cusparseCheck((val), #val, __FILE__, __LINE__)

struct IsNull {
    template<typename T> __device__ bool operator()(const T* p) const {
        return p == nullptr;
    };

    __device__ bool operator()(int index) const {
        return index < 0;
    };

    __device__ bool operator()(const PairNi& p) const {
        return p.first == nullptr;
    };

    __device__ bool operator()(const PairEi& p) const {
        return p.first == nullptr;
    };

    __device__ bool operator()(const PairFi& p) const {
        return p.first == nullptr;
    };

    __device__ bool operator()(const Impact& impact) const {
        return impact.t < 0.0f;
    };

    __device__ bool operator()(const Intersection& intersection) const {
        return intersection.face0 == nullptr && intersection.face1 == nullptr;
    };

    __device__ bool operator()(const Proximity& proximity) const {
        return proximity.stiffness < 0.0f;
    };
};

const int GRID_SIZE = 32;
const int BLOCK_SIZE = 256;

void cudaCheck(cudaError_t err, const char* func, const char* file, int line);
void cudaCheckLast(const char* file, int line);
void cublasCheck(cublasStatus_t err, const char* func, const char* file, int line);
void cusparseCheck(cusparseStatus_t err, const char* func, const char* file, int line);

template<typename T> static T* pointer(thrust::device_vector<T>& v, int offset = 0) {
    return thrust::raw_pointer_cast(v.data() + offset);
}

template<typename T> static const T* pointer(const thrust::device_vector<T>& v, int offset = 0) {
    return thrust::raw_pointer_cast(v.data() + offset);
}

#endif