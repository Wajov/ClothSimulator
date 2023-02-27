#include "CudaHelper.cuh"

void cudaCheck(cudaError_t err, const char* func, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    }
}

void cudaCheckLast(const char* file, int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
    }
}

void cusparseCheck(cusparseStatus_t err, const char* func, const char* file, int line) {
    if (err != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "CuSPARSE Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cusparseGetErrorString(err) << " " << func << std::endl;
    }
}

void cusolverCheck(cusolverStatus_t err, const char* func, const char* file, int line) {
    if (err != CUSOLVER_STATUS_SUCCESS)
        std::cerr << "CuSOLVER Runtime Error at: " << file << ":" << line << std::endl;
}