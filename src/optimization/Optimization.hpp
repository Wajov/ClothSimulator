#ifndef OPTIMIZATION_HPP
#define OPTIMIZATION_HPP

#include <vector>

#include "Vector.cuh"

class Optimization {
protected:
    int nodeSize, constraintSize;

public:
    Optimization();
    virtual ~Optimization();
    int getNodeSize() const;
    int getConstraintSize() const;
    virtual void initialize(std::vector<Vector3f>& x) const = 0;
    virtual void finalize(const std::vector<Vector3f>& x) = 0;
    virtual float objective(const std::vector<Vector3f>& x) const = 0;
    virtual void objectiveGradient(const std::vector<Vector3f>& x, std::vector<Vector3f>& gradient) const = 0;
    virtual float constraint(const std::vector<Vector3f>& x, int index, int& sign) const = 0;
    virtual void constraintGradient(const std::vector<Vector3f>& x, int index, float factor, std::vector<Vector3f>& gradient) const = 0;
};

#endif