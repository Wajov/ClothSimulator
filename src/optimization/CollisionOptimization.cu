#include "CollisionOptimization.cuh"

CollisionOptimization::CollisionOptimization(const std::vector<Impact>& impacts, float thickness, int deform, float obstacleMass) :
    impacts(impacts),
    thickness(thickness),
    obstacleMass(obstacleMass) {
    nConstraints = impacts.size();
    std::vector<Pairni> impactNodes;
    for (int i = 0; i < impacts.size(); i++) {
        const Impact& impact = impacts[i];
        for (int j = 0; j < 4; j++) {
            Node* node = impact.nodes[j];
            if (deform == 1 || node->isFree)
                impactNodes.emplace_back(node, 4 * i + j);
        }
    }

    std::sort(impactNodes.begin(), impactNodes.end());
    invMass = 0.0f;
    indices.assign(4 * nConstraints, -1);
    int index = -1;
    for (int i = 0; i < impactNodes.size(); i++) {
        Node* node = impactNodes[i].first;
        int idx = impactNodes[i].second;
        if (i == 0 || node != impactNodes[i - 1].first) {
            float mass = node->isFree ? node->mass : obstacleMass;
            invMass += 1.0f / mass;
            nodes.push_back(node);
            index++;
        }
        indices[idx] = index;
    }
    nNodes = nodes.size();
    invMass /= nNodes;
}

CollisionOptimization::CollisionOptimization(const thrust::device_vector<Impact>& impacts, float thickness, int deform, float obstacleMass) :
    impactsGpu(impacts),
    thickness(thickness),
    obstacleMass(obstacleMass) {
    nConstraints = impacts.size();
    nodesGpu.resize(4 * nConstraints);
    Node** nodesPointer = pointer(nodesGpu);
    thrust::device_vector<int> nodeIndices(4 * nConstraints);
    int* nodeIndicesPointer = pointer(nodeIndices);
    collectCollisionNodes<<<GRID_SIZE, BLOCK_SIZE>>>(nConstraints, pointer(impacts), deform, nodeIndicesPointer, nodesPointer);
    CUDA_CHECK_LAST();

    nodesGpu.erase(thrust::remove(nodesGpu.begin(), nodesGpu.end(), nullptr), nodesGpu.end());
    nodeIndices.erase(thrust::remove(nodeIndices.begin(), nodeIndices.end(), -1), nodeIndices.end());
    thrust::sort_by_key(nodesGpu.begin(), nodesGpu.end(), nodeIndices.begin());
    thrust::device_vector<int> diff(nodesGpu.size());
    setDiff<<<GRID_SIZE, BLOCK_SIZE>>>(nodesGpu.size(), nodesPointer, pointer(diff));
    CUDA_CHECK_LAST();

    thrust::inclusive_scan(diff.begin(), diff.end(), diff.begin());
    indicesGpu.assign(4 * nConstraints, -1);
    setIndices<<<GRID_SIZE, BLOCK_SIZE>>>(nodesGpu.size(), nodeIndicesPointer, pointer(diff), pointer(indicesGpu));
    CUDA_CHECK_LAST();

    nodesGpu.erase(thrust::unique(nodesGpu.begin(), nodesGpu.end()), nodesGpu.end());
    nNodes = nodesGpu.size();
    thrust::device_vector<float> inv(nNodes);
    collisionInv<<<GRID_SIZE, BLOCK_SIZE>>>(nNodes, nodesPointer, obstacleMass, pointer(inv));
    CUDA_CHECK_LAST();

    invMass = thrust::reduce(inv.begin(), inv.end());
}

CollisionOptimization::~CollisionOptimization() {}

float CollisionOptimization::objective(const std::vector<Vector3f>& x) const {
    float ans = 0.0f;
    for (int i = 0; i < nNodes; i++) {
        Node* node = nodes[i];
        float mass = node->isFree ? node->mass : obstacleMass;
        ans += mass * (x[i] - node->x1).norm2();
    }
    return 0.5f * ans * invMass;
}

float CollisionOptimization::objective(const thrust::device_vector<Vector3f>& x) const {
    thrust::device_vector<float> objectives(nNodes);
    collisionObjective<<<GRID_SIZE, BLOCK_SIZE>>>(nNodes, pointer(nodesGpu), obstacleMass, pointer(x), pointer(objectives));
    CUDA_CHECK_LAST();
    return 0.5f * invMass * thrust::reduce(objectives.begin(), objectives.end());
}

void CollisionOptimization::objectiveGradient(const std::vector<Vector3f>& x, std::vector<Vector3f>& gradient) const {
    for (int i = 0; i < nNodes; i++) {
        Node* node = nodes[i];
        float mass = node->isFree ? node->mass : obstacleMass;
        gradient[i] = invMass * mass * (x[i] - node->x1);
    }
}

void CollisionOptimization::objectiveGradient(const thrust::device_vector<Vector3f>& x, thrust::device_vector<Vector3f>& gradient) const {
    collisionObjectiveGradient<<<GRID_SIZE, BLOCK_SIZE>>>(nNodes, pointer(nodesGpu), invMass, obstacleMass, pointer(x), pointer(gradient));
    CUDA_CHECK_LAST();
}

float CollisionOptimization::constraint(const std::vector<Vector3f>& x, int index, int& sign) const {
    sign = 1;
    float ans = -thickness;
    const Impact& impact = impacts[index];
    for (int i = 0; i < 4; i++) {
        int j = indices[4 * index + i];
        if (j > -1)
            ans += impact.w[i] * impact.n.dot(x[j]);
        else
            ans += impact.w[i] * impact.n.dot(impact.nodes[i]->x);
    }
    return ans;
}

void CollisionOptimization::constraint(const thrust::device_vector<Vector3f>& x, thrust::device_vector<float>& constraints, thrust::device_vector<int>& signs) const {
    collisionConstraint<<<GRID_SIZE, BLOCK_SIZE>>>(nConstraints, pointer(impactsGpu), pointer(indicesGpu), thickness, pointer(x), pointer(constraints), pointer(signs));
    CUDA_CHECK_LAST();
}

void CollisionOptimization::constraintGradient(const std::vector<Vector3f>& x, int index, float factor, std::vector<Vector3f>& gradient) const {
    const Impact& impact = impacts[index];
    for (int i = 0; i < 4; i++) {
        int j = indices[4 * index + i];
        if (j > -1)
            gradient[j] += factor * impact.w[i] * impact.n;
    }
}

void CollisionOptimization::constraintGradient(const thrust::device_vector<Vector3f>& x, const thrust::device_vector<float>& coefficients, float mu, thrust::device_vector<Vector3f>& gradient) const {
    thrust::device_vector<int> gradIndices = indicesGpu;
    thrust::device_vector<Vector3f> grad(4 * nConstraints);
    collectCollisionConstraintGradient<<<GRID_SIZE, BLOCK_SIZE>>>(nConstraints, pointer(impactsGpu), pointer(coefficients), mu, pointer(grad));
    CUDA_CHECK_LAST();

    grad.erase(thrust::remove_if(grad.begin(), grad.end(), gradIndices.begin(), IsNull()), grad.end());
    gradIndices.erase(thrust::remove(gradIndices.begin(), gradIndices.end(), -1), gradIndices.end());
    thrust::sort_by_key(gradIndices.begin(), gradIndices.end(), grad.begin());
    thrust::device_vector<int> outputGradIndices(4 * nConstraints);
    thrust::device_vector<Vector3f> outputGrad(4 * nConstraints);
    auto iter = thrust::reduce_by_key(gradIndices.begin(), gradIndices.end(), grad.begin(), outputGradIndices.begin(), outputGrad.begin());
    addConstraintGradient<<<GRID_SIZE, BLOCK_SIZE>>>(iter.first - outputGradIndices.begin(), pointer(outputGradIndices), pointer(outputGrad), pointer(gradient));
    CUDA_CHECK_LAST();
}
