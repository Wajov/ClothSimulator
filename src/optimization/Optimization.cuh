#ifndef OPTIMIZATION_CUH
#define OPTIMIZATION_CUH

#include <vector>

#include "MathHelper.cuh"
#include "Vector.cuh"

const int MAX_ITERATIONS = 100;
const float EPSILON = 1e-12f;
const float RHO = 0.9992f;
const float RHO2 = RHO * RHO;

class Optimization {
protected:
    int nNodes, nConstraints;
    float mu;
    std::vector<float> lambda;
    float clampViolation(float x, int sign) const;
    float value(const std::vector<Vector3f>& x) const;
    void valueAndGradient(const std::vector<Vector3f>& x, float& value, std::vector<Vector3f>& gradient) const;
    void updateMultiplier(const std::vector<Vector3f>& x);

public:
    Optimization();
    virtual ~Optimization();
    virtual void initialize(std::vector<Vector3f>& x) const = 0;
    virtual void finalize(const std::vector<Vector3f>& x) = 0;
    virtual float objective(const std::vector<Vector3f>& x) const = 0;
    virtual void objectiveGradient(const std::vector<Vector3f>& x, std::vector<Vector3f>& gradient) const = 0;
    virtual float constraint(const std::vector<Vector3f>& x, int index, int& sign) const = 0;
    virtual void constraintGradient(const std::vector<Vector3f>& x, int index, float factor, std::vector<Vector3f>& gradient) const = 0;
    void solve();
};

#endif