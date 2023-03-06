#include "Optimization.cuh"

Optimization::Optimization() {}

Optimization::~Optimization() {}

void Optimization::initialize(std::vector<Vector3f>& x) const {
    for (int i = 0; i < nNodes; i++)
        x[i] = nodes[i]->x;
}

void Optimization::initialize(thrust::device_vector<Vector3f>& x) const {
    initializeGpu<<<GRID_SIZE, BLOCK_SIZE>>>(nNodes, pointer(nodesGpu), pointer(x));
}

void Optimization::finalize(const std::vector<Vector3f>& x) {
    for (int i = 0; i < nNodes; i++)
        nodes[i]->x = x[i];
}

void Optimization::finalize(const thrust::device_vector<Vector3f>& x) {
    finalizeGpu<<<GRID_SIZE, BLOCK_SIZE>>>(nNodes, pointer(x), pointer(nodesGpu));
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

float Optimization::value(const thrust::device_vector<Vector3f>& x) const {
    thrust::device_vector<float> c(nConstraints);
    thrust::device_vector<int> signs(nConstraints);
    constraint(x, c, signs);
    computeCoefficient<<<GRID_SIZE, BLOCK_SIZE>>>(nConstraints, pointer(lambdaGpu), mu, pointer(signs), pointer(c));
    CUDA_CHECK_LAST();

    computeSquare<<<GRID_SIZE, BLOCK_SIZE>>>(nConstraints, pointer(c));
    CUDA_CHECK_LAST();

    return objective(x) + 0.5 * mu * thrust::reduce(c.begin(), c.end());
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

void Optimization::valueAndGradient(const thrust::device_vector<Vector3f>& x, float& value, thrust::device_vector<Vector3f>& gradient) const {
    objectiveGradient(x, gradient);

    thrust::device_vector<float> c(nConstraints);
    thrust::device_vector<int> signs(nConstraints);
    constraint(x, c, signs);
    computeCoefficient<<<GRID_SIZE, BLOCK_SIZE>>>(nConstraints, pointer(lambdaGpu), mu, pointer(signs), pointer(c));
    CUDA_CHECK_LAST();

    constraintGradient(x, c, mu, gradient);
    
    computeSquare<<<GRID_SIZE, BLOCK_SIZE>>>(nConstraints, pointer(c));
    CUDA_CHECK_LAST();

    value = objective(x) + 0.5 * mu * thrust::reduce(c.begin(), c.end());
}

void Optimization::updateMultiplier(const std::vector<Vector3f>& x) {
    for (int i = 0; i < nConstraints; i++) {
        int sign;
        float c = constraint(x, i, sign);
        lambda[i] = clampViolation(lambda[i] + mu * c, sign);
    }
}

void Optimization::updateMultiplier(const thrust::device_vector<Vector3f>& x) {
    thrust::device_vector<float> c(nConstraints);
    thrust::device_vector<int> signs(nConstraints);
    constraint(x, c, signs);
    updateMultiplierGpu<<<GRID_SIZE, BLOCK_SIZE>>>(nConstraints, pointer(c), pointer(signs), mu, pointer(lambdaGpu));
    CUDA_CHECK_LAST();
}

void Optimization::solve() {
    float s = 1e-3f, omega = 1.0f, f;
    mu = 1e3f;
    if (!gpu) {
        std::vector<Vector3f> x(nNodes), gradient(nNodes), xt(nNodes);
        lambda.assign(nConstraints, 0.0f);
        initialize(x);

        for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
            valueAndGradient(x, f, gradient);
        
            float norm2 = 0.0f;
            for (int i = 0; i < nNodes; i++)
                norm2 += gradient[i].norm2();
            s /= 0.7f;
            do {
                s *= 0.7f;
                for (int i = 0; i < nNodes; i++)
                    xt[i] = x[i] - s * gradient[i];
            } while (value(xt) >= f - 0.5f * s * norm2 && s >= EPSILON);
            if (s < EPSILON)
                break;

            if (iter == 10)
                omega = 2.0f / (2.0f - RHO2);
            else if (iter > 10)
                omega = 4.0f / (4.0f - RHO2 * omega);
            float coeffient = (1 + omega) * s;
            for (int i = 0; i < nNodes; i++)
                x[i] -= coeffient * gradient[i];
            
            updateMultiplier(x);
        }
        finalize(x);
    } else {
        thrust::device_vector<Vector3f> x(nNodes), gradient(nNodes), xt(nNodes);
        thrust::device_vector<float> gradient2(nNodes);
        Vector3f* xPointer = pointer(x);
        Vector3f* gradientPointer = pointer(gradient);
        Vector3f* xtPointer = pointer(xt);
        float* gradient2Pointer = pointer(gradient2);
        lambdaGpu.assign(nConstraints, 0.0f);
        initialize(x);

        for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
            valueAndGradient(x, f, gradient);

            computeNorm2<<<GRID_SIZE, BLOCK_SIZE>>>(nNodes, gradientPointer, gradient2Pointer);
            float norm2 = thrust::reduce(gradient2.begin(), gradient2.end());
            s /= 0.7f;
            do {
                s *= 0.7f;
                computeXt<<<GRID_SIZE, BLOCK_SIZE>>>(nNodes, xPointer, gradientPointer, s, xtPointer);
            } while (value(xt) >= f - 0.5f * s * norm2 && s >= EPSILON);
            if (s < EPSILON)
                break;
            
            if (iter == 10)
                omega = 2.0f / (2.0f - RHO2);
            else if (iter > 10)
                omega = 4.0f / (4.0f - RHO2 * omega);
            float coeffient = (1 + omega) * s;
            updateX<<<GRID_SIZE, BLOCK_SIZE>>>(nNodes, gradientPointer, coeffient, xPointer);

            updateMultiplier(x);
        }
        finalize(x);
    }
}