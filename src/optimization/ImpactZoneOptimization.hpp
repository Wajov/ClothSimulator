#ifndef IMPACT_ZONE_OPTIMIZATION_HPP
#define IMPACT_ZONE_OPTIMIZATION_HPP

#include <vector>

#include "MathHelper.cuh"
#include "Optimization.hpp"
#include "Node.cuh"
#include "Impact.hpp"
#include "ImpactZone.hpp"

class ImpactZoneOptimization : public Optimization {
private:
    double invMass, thickness, obstacleMass;
    std::vector<Node*> nodes;
    std::vector<Impact> impacts;
    std::vector<std::vector<int>> indices;

public:
    ImpactZoneOptimization(const ImpactZone* zone, double thickness, double obstacleMass);
    ~ImpactZoneOptimization();
    void initialize(double* x) const override;
    void precompute(const double *x) override;
    void finalize(const double* x) override;
    double objective(const double* x) const override;
    void objectiveGradient(const double* x, double* gradient) const override;
    double constraint(const double* x, int index, int& sign) const override;
    void constraintGradient(const double* x, int index, double factor, double* gradient) const override;
};

#endif