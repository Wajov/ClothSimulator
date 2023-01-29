#ifndef CUDA_HELPER_CUH
#define CUDA_HELPER_CUH

#include <iostream>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <thrust/pair.h>
#include <thrust/device_vector.h>

#define CUDA_CHECK(val) cudaCheck((val), #val, __FILE__, __LINE__)
#define CUDA_CHECK_LAST() cudaCheckLast(__FILE__, __LINE__)
#define CUSPARSE_CHECK(val) cusparseCheck((val), #val, __FILE__, __LINE__)
#define CUSOLVER_CHECK(val) cusolverCheck((val), #val, __FILE__, __LINE__)

typedef thrust::pair<int, int> PairIndex;

const int GRID_SIZE = 32;
const int BLOCK_SIZE = 256;

static void cudaCheck(cudaError_t err, const char* func, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    }
}

static void cudaCheckLast(const char* file, int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
    }
}

static void cusparseCheck(cusparseStatus_t err, const char* func, const char* file, int line) {
    if (err != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "CuSPARSE Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cusparseGetErrorString(err) << " " << func << std::endl;
    }
}

static void cusolverCheck(cusolverStatus_t err, const char* func, const char* file, int line) {
    if (err != CUSOLVER_STATUS_SUCCESS)
        std::cerr << "CuSOLVER Runtime Error at: " << file << ":" << line << std::endl;
}

template<typename T> static T* pointer(thrust::device_vector<T>& v, int offset = 0) {
    return thrust::raw_pointer_cast(v.data() + offset);
}

#endif