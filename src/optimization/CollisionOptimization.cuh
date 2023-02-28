#ifndef COLLISION_OPTIMIZATION_CUH
#define COLLISION_OPTIMIZATION_CUH

#include <algorithm>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>

#include "MathHelper.cuh"
#include "CudaHelper.cuh"
#include "Optimization.cuh"
#include "Pair.cuh"
#include "Node.cuh"
#include "Impact.cuh"

class CollisionOptimization : public Optimization {
private:
    float invMass, thickness, obstacleMass;
    std::vector<Impact> impacts;
    std::vector<int> indices;
    thrust::device_vector<Impact> impactsGpu;
    thrust::device_vector<int> indicesGpu;

protected:
    float objective(const std::vector<Vector3f>& x) const override;
    float objective(const thrust::device_vector<Vector3f>& x) const override;
    void objectiveGradient(const std::vector<Vector3f>& x, std::vector<Vector3f>& gradient) const override;
    void objectiveGradient(const thrust::device_vector<Vector3f>& x, thrust::device_vector<Vector3f>& gradient) const override;
    float constraint(const std::vector<Vector3f>& x, int index, int& sign) const override;
    void constraint(const thrust::device_vector<Vector3f>& x, thrust::device_vector<float>& constraints, thrust::device_vector<int>& signs) const override;
    void constraintGradient(const std::vector<Vector3f>& x, int index, float factor, std::vector<Vector3f>& gradient) const override;
    void constraintGradient(const thrust::device_vector<Vector3f>& x, const thrust::device_vector<float>& coefficients, float mu, thrust::device_vector<Vector3f>& gradient) const override;

public:
    CollisionOptimization(const std::vector<Impact>& impacts, float thickness, int deform, float obstacleMass);
    CollisionOptimization(const thrust::device_vector<Impact>& impacts, float thickness, int deform, float obstacleMass);
    ~CollisionOptimization();
};

#endif