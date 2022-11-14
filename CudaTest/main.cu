#include <vector>
#include <random>
#include <chrono>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>

struct Ref {
    int indices[144];
};

__global__ static void sum(int n, const Ref* d, int* s) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = gridDim.x * blockIdx.x + threadIdx.x; i < n; i += nThreads)
        for (int j = 0; j < 144; j++)
            atomicAdd(&s[d[i].indices[j]], 1);
}

__global__ static void collect(int n, const Ref* d, int* indices, int* values) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = gridDim.x * blockIdx.x + threadIdx.x; i < 144 * n; i += nThreads) {
        int j = i / 144, k = i % 144;
        indices[i] = d[j].indices[k];
        values[i] = 1;
    }
}

__global__ static void set(int n, const int* indices, const int* values, int* s) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = gridDim.x * blockIdx.x + threadIdx.x; i < n; i += nThreads)
        s[indices[i]] = values[i];
}

int main() {
    int n = 100000, m = 10000;
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distribution(0, m - 1);

    std::vector<Ref> h(n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < 144; j++)
            h[i].indices[j] = distribution(gen);

    std::chrono::duration<float> dt;
    auto t0 = std::chrono::high_resolution_clock::now();
    
    // CPU
    std::vector<int> s(m, 0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < 144; j++)
            s[h[i].indices[j]]++;

    auto t1 = std::chrono::high_resolution_clock::now();
    dt = t1 - t0;
    std::cout << "CPU: " << dt.count() << "s" << std::endl;

    // GPU: atomicAdd
    thrust::device_vector<Ref> d = h;
    thrust::device_vector<int> sGpu0(m, 0);
    sum<<<256, 256>>>(d.size(), thrust::raw_pointer_cast(d.data()), thrust::raw_pointer_cast(sGpu0.data()));
    cudaDeviceSynchronize();

    auto t2 = std::chrono::high_resolution_clock::now();
    dt = t2 - t1;
    std::cout << "GPU(atomicAdd): " << dt.count() << "s" << std::endl;

    thrust::device_vector<int> indices(144 * n);
    thrust::device_vector<int> values(144 * n);
    collect<<<256, 256>>>(d.size(), thrust::raw_pointer_cast(d.data()), thrust::raw_pointer_cast(indices.data()), thrust::raw_pointer_cast(values.data()));
    thrust::sort_by_key(indices.begin(), indices.end(), values.begin());
    thrust::device_vector<int> outputIndices(144 * n);
    thrust::device_vector<int> outputValues(144 * n);
    auto iter = thrust::reduce_by_key(indices.begin(), indices.end(), values.begin(), outputIndices.begin(), outputValues.begin());
    thrust::device_vector<int> sGpu1(m, 0);
    set<<<256, 256>>>(iter.first - outputIndices.begin(), thrust::raw_pointer_cast(outputIndices.data()), thrust::raw_pointer_cast(outputValues.data()), thrust::raw_pointer_cast(sGpu1.data()));cudaDeviceSynchronize();

    auto t3 = std::chrono::high_resolution_clock::now();
    dt = t3 - t2;
    std::cout << "GPU(sort and reduce): " << dt.count() << "s" << std::endl;

    thrust::host_vector<int> s0 = sGpu0;
    thrust::host_vector<int> s1 = sGpu1;

    for (int i = 0; i < m; i++)
        if (s[i] != s0[i] || s[i] != s1[i])
            std::cout << i << ": " << s[i] << " " << s0[i] << " " << s1[i] << std::endl;

    return 0;
}