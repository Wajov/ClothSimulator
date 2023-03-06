#ifndef PAIR_CUH
#define PAIR_CUH

#include <cuda_runtime.h>

#include "Matrix.cuh"
#include "Node.cuh"
#include "Edge.cuh"
#include "Face.cuh"

template<typename S, typename T> class Pair {
public:
    S first;
    T second;

    __host__ __device__ Pair() :
        first(static_cast<S>(0)),
        second(static_cast<T>(0)) {};
    
    __host__ __device__ Pair(S first, T second) :
        first(first),
        second(second) {};

    __host__ __device__ Pair(const Pair<S, T>& p) :
        first(p.first),
        second(p.second) {};
    
    __host__ __device__ ~Pair() {};

    __host__ __device__ bool operator==(const Pair<S, T>& p) const {
        return first == p.first && second == p.second;
    };

    __host__ __device__ bool operator!=(const Pair<S, T>& p) const {
        return first != p.first || second != p.second;
    };

    __host__ __device__ bool operator<(const Pair<S, T>& p) const {
        return first < p.first || first == p.first && second < p.second;
    };

    __host__ __device__ Pair<S, T> operator+(const Pair<S, T>& p) const {
        Pair<S, T> ans(first + p.first, second + p.second);
        return ans;
    }
};

typedef Pair<int, int> Pairii;
typedef Pair<float, int> Pairfi;
typedef Pair<float, Matrix2x2f> Pairfm;
typedef Pair<Node*, int> Pairni;
typedef Pair<Edge*, int> Pairei;
typedef Pair<float, Edge*> Pairfe;
typedef Pair<Face*, Face*> Proximity;

#endif