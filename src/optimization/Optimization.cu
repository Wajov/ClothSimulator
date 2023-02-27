#include "Optimization.cuh"

Optimization::Optimization() {}

Optimization::~Optimization() {}

float Optimization::clampViolation(float x, int sign) const {
    return sign < 0 ? max(x, 0.0f) : (sign > 0 ? min(x, 0.0f) : x);
}

float Optimization::value(const std::vector<Vector3f>& x) const {
    float ans = objective(x);
    for (int i = 0; i < nConstraints; i++) {
        int sign;
        float c = constraint(x, i, sign);
        float coefficient = clampViolation(c + lambda[i] / mu, sign);
        if (coefficient != 0.0f)
            ans += 0.5f * mu * sqr(coefficient);
    }
    return ans;
}

void Optimization::valueAndGradient(const std::vector<Vector3f>& x, float& value, std::vector<Vector3f>& gradient) const {
    value = objective(x);
    objectiveGradient(x, gradient);

    for (int i = 0; i < nConstraints; i++) {
        int sign;
        float c = constraint(x, i, sign);
        float coefficient = clampViolation(c + lambda[i] / mu, sign);
        if (coefficient != 0.0f) {
            value += 0.5f * mu * sqr(coefficient);
            constraintGradient(x, i, mu * coefficient, gradient);
        }
    }
}

void Optimization::updateMultiplier(const std::vector<Vector3f>& x) {
    for (int i = 0; i < nConstraints; i++) {
        int sign;
        float c = constraint(x, i, sign);
        lambda[i] = clampViolation(lambda[i] + mu * c, sign);
    }
}

void Optimization::solve() {
    float s = 1e-3f, omega = 1.0f, f;
    std::vector<Vector3f> x(nNodes), gradient(nNodes), t(nNodes);
    mu = 1e3f;
    lambda.assign(nConstraints, 0.0f);
    initialize(x);

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        valueAndGradient(x, f, gradient);
       
        float norm2 = 0;
        for (int i = 0; i < nNodes; i++)
            norm2 += gradient[i].norm2();
        s /= 0.7f;
        do {
            s *= 0.7f;
            for (int i = 0; i < nNodes; i++)
                t[i] = x[i] - s * gradient[i];
        } while (value(t) >= f - 0.5f * s * norm2 && s >= EPSILON);
        if (s < EPSILON)
            break;

        if (iter == 10)
            omega = 2.0f / (2.0f - RHO2);
        else if (iter > 10)
            omega = 4.0f / (4.0f - RHO2 * omega);
        float coeffient = (1 + omega) * s;
        for (int i = 0; i < nNodes; i++)
            x[i] = x[i] - coeffient * gradient[i];
        
        updateMultiplier(x);
    }
    finalize(x);
}