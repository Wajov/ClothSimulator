#ifndef COLLISION_OPTIMIZATION_CUH
#define COLLISION_OPTIMIZATION_CUH

#include <algorithm>
#include <vector>

#include "MathHelper.cuh"
#include "Optimization.cuh"
#include "Pair.cuh"
#include "Node.cuh"
#include "Impact.cuh"

class CollisionOptimization : public Optimization {
private:
    float invMass, thickness, obstacleMass;
    std::vector<Node*> nodes;
    std::vector<Impact> impacts;
    std::vector<std::vector<int>> indices;

public:
    CollisionOptimization(const std::vector<Impact>& impacts, float thickness, int deform, float obstacleMass);
    ~CollisionOptimization();
    void initialize(std::vector<Vector3f>& x) const override;
    void finalize(const std::vector<Vector3f>& x) override;
    float objective(const std::vector<Vector3f>& x) const override;
    void objectiveGradient(const std::vector<Vector3f>& x, std::vector<Vector3f>& gradient) const override;
    float constraint(const std::vector<Vector3f>& x, int index, int& sign) const override;
    void constraintGradient(const std::vector<Vector3f>& x, int index, float factor, std::vector<Vector3f>& gradient) const override;
};

#endif