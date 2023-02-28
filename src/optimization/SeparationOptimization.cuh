#ifndef SEPARATION_OPTIMIZATION_CUH
#define SEPARATION_OPTIMIZATION_CUH

#include <vector>

#include "MathHelper.cuh"
#include "Optimization.cuh"
#include "Node.cuh"
#include "Intersection.cuh"

class SeparationOptimization : public Optimization {
private:
    float invArea, thickness;
    std::vector<Intersection> intersections;
    std::vector<std::vector<int>> indices;
    int addNode(const Node* node);

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
    SeparationOptimization(const std::vector<Intersection>& intersections, float thickness);
    ~SeparationOptimization();
};

#endif