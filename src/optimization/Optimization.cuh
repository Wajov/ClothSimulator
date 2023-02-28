#ifndef OPTIMIZATION_CUH
#define OPTIMIZATION_CUH

#include <vector>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "MathHelper.cuh"
#include "CudaHelper.cuh"
#include "OptimizationHelper.cuh"
#include "Vector.cuh"
#include "Node.cuh"

const int MAX_ITERATIONS = 100;
const float EPSILON = 1e-12f;
const float RHO = 0.9992f;
const float RHO2 = RHO * RHO;

extern bool gpu;

class Optimization {
protected:
    int nNodes, nConstraints;
    float mu;
    std::vector<Node*> nodes;
    std::vector<float> lambda;
    thrust::device_vector<Node*> nodesGpu;
    thrust::device_vector<float> lambdaGpu;
    void initialize(std::vector<Vector3f>& x) const;
    void initialize(thrust::device_vector<Vector3f>& x) const;
    void finalize(const std::vector<Vector3f>& x);
    void finalize(const thrust::device_vector<Vector3f>& x);
    virtual float objective(const std::vector<Vector3f>& x) const = 0;
    virtual float objective(const thrust::device_vector<Vector3f>& x) const = 0;
    virtual void objectiveGradient(const std::vector<Vector3f>& x, std::vector<Vector3f>& gradient) const = 0;
    virtual void objectiveGradient(const thrust::device_vector<Vector3f>& x, thrust::device_vector<Vector3f>& gradient) const = 0;
    virtual float constraint(const std::vector<Vector3f>& x, int index, int& sign) const = 0;
    virtual void constraint(const thrust::device_vector<Vector3f>& x, thrust::device_vector<float>& constraints, thrust::device_vector<int>& signs) const = 0;
    virtual void constraintGradient(const std::vector<Vector3f>& x, int index, float factor, std::vector<Vector3f>& gradient) const = 0;
    virtual void constraintGradient(const thrust::device_vector<Vector3f>& x, const thrust::device_vector<float>& coefficients, float mu, thrust::device_vector<Vector3f>& gradient) const = 0;
    float value(const std::vector<Vector3f>& x) const;
    float value(const thrust::device_vector<Vector3f>& x) const;
    void valueAndGradient(const std::vector<Vector3f>& x, float& value, std::vector<Vector3f>& gradient) const;
    void valueAndGradient(const thrust::device_vector<Vector3f>& x, float& value, thrust::device_vector<Vector3f>& gradient) const;
    void updateMultiplier(const std::vector<Vector3f>& x);
    void updateMultiplier(const thrust::device_vector<Vector3f>& x);

public:
    Optimization();
    virtual ~Optimization();
    void solve();
};

#endif