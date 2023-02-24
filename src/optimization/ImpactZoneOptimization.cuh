#ifndef IMPACT_ZONE_OPTIMIZATION_CUH
#define IMPACT_ZONE_OPTIMIZATION_CUH

#include <vector>

#include "MathHelper.cuh"
#include "Optimization.cuh"
#include "Node.cuh"
#include "Impact.cuh"
#include "ImpactZone.cuh"

class ImpactZoneOptimization : public Optimization {
private:
    float invMass, thickness, obstacleMass;
    std::vector<Node*> nodes;
    std::vector<Impact> impacts;
    std::vector<std::vector<int>> indices;

public:
    ImpactZoneOptimization(const ImpactZone* zone, float thickness, float obstacleMass);
    ~ImpactZoneOptimization();
    void initialize(std::vector<Vector3f>& x) const override;
    void finalize(const std::vector<Vector3f>& x) override;
    float objective(const std::vector<Vector3f>& x) const override;
    void objectiveGradient(const std::vector<Vector3f>& x, std::vector<Vector3f>& gradient) const override;
    float constraint(const std::vector<Vector3f>& x, int index, int& sign) const override;
    void constraintGradient(const std::vector<Vector3f>& x, int index, float factor, std::vector<Vector3f>& gradient) const override;
};

#endif