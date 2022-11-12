#ifndef CUDA_HELPER_CUH
#define CUDA_HELPER_CUH

#include <iostream>

#include <cuda_runtime.h>
#include <thrust/pair.h>
#include <thrust/device_vector.h>

#define CUDA_CHECK(val) cudaCheck((val), #val, __FILE__, __LINE__)
#define CUDA_CHECK_LAST() cudaCheckLast(__FILE__, __LINE__)

typedef thrust::pair<int, int> PairIndex;

static const int GRID_SIZE = 32;
static const int BLOCK_SIZE = 256;

template <typename T> static void cudaCheck(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    }
}

static void cudaCheckLast(const char* const file, const int line) {
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
    }
}

template<typename T> static T* pointer(thrust::device_vector<T>& v, int offset = 0) {
    return thrust::raw_pointer_cast(v.data() + offset);
}

#endif