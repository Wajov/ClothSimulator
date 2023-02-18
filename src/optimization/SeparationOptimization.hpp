#ifndef SEPARATION_OPTIMIZATION_HPP
#define SEPARATION_OPTIMIZATION_HPP

#include <vector>

#include "MathHelper.cuh"
#include "Optimization.hpp"
#include "Node.cuh"
#include "Intersection.hpp"

class SeparationOptimization : public Optimization {
private:
    double invArea, thickness;
    std::vector<Node*> nodes;
    std::vector<Intersection> intersections;
    std::vector<std::vector<int>> indices;
    int addNode(const Node* node);

public:
    SeparationOptimization(const std::vector<Intersection>& intersections, double thickness);
    ~SeparationOptimization();
    void initialize(double* x) const override;
    void precompute(const double *x) override;
    void finalize(const double* x) override;
    double objective(const double* x) const override;
    void objectiveGradient(const double* x, double* gradient) const override;
    double constraint(const double* x, int index, int& sign) const override;
    void constraintGradient(const double* x, int index, double factor, double* gradient) const override;
};

#endif