#ifndef SEPARATION_OPTIMIZATION_HPP
#define SEPARATION_OPTIMIZATION_HPP

#include <vector>

#include "MathHelper.cuh"
#include "Optimization.hpp"
#include "Node.cuh"
#include "Intersection.hpp"

class SeparationOptimization : public Optimization {
private:
    float invArea, thickness;
    std::vector<Node*> nodes;
    std::vector<Intersection> intersections;
    std::vector<std::vector<int>> indices;
    int addNode(const Node* node);

public:
    SeparationOptimization(const std::vector<Intersection>& intersections, float thickness);
    ~SeparationOptimization();
    void initialize(std::vector<Vector3f>& x) const override;
    void finalize(const std::vector<Vector3f>& x) override;
    float objective(const std::vector<Vector3f>& x) const override;
    void objectiveGradient(const std::vector<Vector3f>& x, std::vector<Vector3f>& gradient) const override;
    float constraint(const std::vector<Vector3f>& x, int index, int& sign) const override;
    void constraintGradient(const std::vector<Vector3f>& x, int index, float factor, std::vector<Vector3f>& gradient) const override;
};

#endif