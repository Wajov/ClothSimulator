#include "SeparationOptimization.cuh"

SeparationOptimization::SeparationOptimization(const std::vector<Intersection>& intersections, float thickness, int deform, float obstacleArea) :
    intersections(intersections),
    thickness(thickness),
    obstacleArea(obstacleArea) {
    nConstraints = intersections.size();
    std::vector<PairNi> intersectionNodes;
    for (int i = 0; i < intersections.size(); i++) {
        const Intersection& intersection = intersections[i];
        for (int j = 0; j < 3; j++) {
            Node* node0 = intersection.face0->vertices[j]->node;
            if (deform == 1 || node0->isFree)
                intersectionNodes.emplace_back(node0, 6 * i + j);
            
            Node* node1 = intersection.face1->vertices[j]->node;
            if (deform == 1 || node1->isFree)
                intersectionNodes.emplace_back(node1, 6 * i + j + 3);
        }
    }

    std::sort(intersectionNodes.begin(), intersectionNodes.end());
    invArea = 0.0f;
    indices.assign(6 * nConstraints, -1);
    int index = -1;
    for (int i = 0; i < intersectionNodes.size(); i++) {
        Node* node = intersectionNodes[i].first;
        int idx = intersectionNodes[i].second;
        if (i == 0 || node != intersectionNodes[i - 1].first) {
            float area = node->isFree ? node->area : obstacleArea;
            invArea += 1.0f / area;
            nodes.push_back(node);
            index++;
        }
        indices[idx] = index;
    }
    nNodes = nodes.size();
    invArea /= nNodes;
}

SeparationOptimization::SeparationOptimization(const thrust::device_vector<Intersection>& intersections, float thickness, int deform, float obstacleArea) :
    intersectionsGpu(intersections),
    thickness(thickness),
    obstacleArea(obstacleArea) {
    nConstraints = intersections.size();
    nodesGpu.resize(6 * nConstraints);
    Node** nodesPointer = pointer(nodesGpu);
    thrust::device_vector<int> nodeIndices(6 * nConstraints);
    int *nodeIndicesPointer = pointer(nodeIndices);
    collectSeparationNodes<<<GRID_SIZE, BLOCK_SIZE>>>(nConstraints, pointer(intersections), deform, nodeIndicesPointer, nodesPointer);
    CUDA_CHECK_LAST();

    nodesGpu.erase(thrust::remove(nodesGpu.begin(), nodesGpu.end(), nullptr), nodesGpu.end());
    nodeIndices.erase(thrust::remove(nodeIndices.begin(), nodeIndices.end(), -1), nodeIndices.end());
    thrust::sort_by_key(nodesGpu.begin(), nodesGpu.end(), nodeIndices.begin());
    thrust::device_vector<int> diff(nodesGpu.size());
    setDiff<<<GRID_SIZE, BLOCK_SIZE>>>(nodesGpu.size(), nodesPointer, pointer(diff));
    CUDA_CHECK_LAST();

    thrust::inclusive_scan(diff.begin(), diff.end(), diff.begin());
    indicesGpu.assign(6 * nConstraints, -1);
    setIndices<<<GRID_SIZE, BLOCK_SIZE>>>(nodesGpu.size(), nodeIndicesPointer, pointer(diff), pointer(indicesGpu));
    CUDA_CHECK_LAST();

    nodesGpu.erase(thrust::unique(nodesGpu.begin(), nodesGpu.end()), nodesGpu.end());
    nNodes = nodesGpu.size();
    thrust::device_vector<float> inv(nNodes);
    separationInv<<<GRID_SIZE, BLOCK_SIZE>>>(nNodes, nodesPointer, obstacleArea, pointer(inv));
    CUDA_CHECK_LAST();

    invArea = thrust::reduce(inv.begin(), inv.end()) / nNodes;
}

SeparationOptimization::~SeparationOptimization() {}

float SeparationOptimization::objective(const std::vector<Vector3f>& x) const {
    float ans = 0.0f;
    for (int i = 0; i < nNodes; i++) {
        Node* node = nodes[i];
        float area = node->isFree ? node->area : obstacleArea;
        ans += area * (x[i] - node->x1).norm2();
    }
    return 0.5f * ans * invArea;
}

float SeparationOptimization::objective(const thrust::device_vector<Vector3f>& x) const {
    thrust::device_vector<float> objectives(nNodes);
    separationObjective<<<GRID_SIZE, BLOCK_SIZE>>>(nNodes, pointer(nodesGpu), obstacleArea, pointer(x), pointer(objectives));
    CUDA_CHECK_LAST();

    return 0.5f * invArea * thrust::reduce(objectives.begin(), objectives.end());
}

void SeparationOptimization::objectiveGradient(const std::vector<Vector3f>& x, std::vector<Vector3f>& gradient) const {
    for (int i = 0; i < nNodes; i++) {
        Node* node = nodes[i];
        float area = node->isFree ? node->area : obstacleArea;
        gradient[i] = invArea * area * (x[i] - node->x1);
    }
}

void SeparationOptimization::objectiveGradient(const thrust::device_vector<Vector3f>& x, thrust::device_vector<Vector3f>& gradient) const {
    separationObjectiveGradient<<<GRID_SIZE, BLOCK_SIZE>>>(nNodes, pointer(nodesGpu), invArea, obstacleArea, pointer(x), pointer(gradient));
    CUDA_CHECK_LAST();
}

float SeparationOptimization::constraint(const std::vector<Vector3f>& x, int index, int& sign) const {
    sign = 1;
    float ans = -thickness;
    const Intersection& intersection = intersections[index];
    for (int i = 0; i < 3; i++) {
        int j0 = indices[6 * index + i];
        if (j0 > -1)
            ans += intersection.b0(i) * intersection.d.dot(x[j0]);
        else
            ans += intersection.b0(i) * intersection.d.dot(intersection.face0->vertices[i]->node->x);

        int j1 = indices[6 * index + i + 3];
        if (j1 > -1)
            ans -= intersection.b1(i) * intersection.d.dot(x[j1]);
        else
            ans -= intersection.b1(i) * intersection.d.dot(intersection.face1->vertices[i]->node->x);
    }
    return ans;
}

void SeparationOptimization::constraint(const thrust::device_vector<Vector3f>& x, thrust::device_vector<float>& constraints, thrust::device_vector<int>& signs) const {
    separationConstraint<<<GRID_SIZE, BLOCK_SIZE>>>(nConstraints, pointer(intersectionsGpu), pointer(indicesGpu), thickness, pointer(x), pointer(constraints), pointer(signs));
    CUDA_CHECK_LAST();
}

void SeparationOptimization::constraintGradient(const std::vector<Vector3f>& x, int index, float factor, std::vector<Vector3f>& gradient) const {
    const Intersection& intersection = intersections[index];
    for (int i = 0; i < 3; i++) {
        int j0 = indices[6 * index + i];
        if (j0 > -1)
            gradient[j0] += factor * intersection.b0(i) * intersection.d;
        
        int j1 = indices[6 * index + i + 3];
        if (j1 > -1)
            gradient[j1] -= factor * intersection.b1(i) * intersection.d;
    }
}

void SeparationOptimization::constraintGradient(const thrust::device_vector<Vector3f>& x, const thrust::device_vector<float>& coefficients, float mu, thrust::device_vector<Vector3f>& gradient) const {
    thrust::device_vector<int> gradIndices = indicesGpu;
    thrust::device_vector<Vector3f> grad(6 * nConstraints);
    collectSeparationConstraintGradient<<<GRID_SIZE, BLOCK_SIZE>>>(nConstraints, pointer(intersectionsGpu), pointer(coefficients), mu, pointer(grad));
    CUDA_CHECK_LAST();

    grad.erase(thrust::remove_if(grad.begin(), grad.end(), gradIndices.begin(), IsNull()), grad.end());
    gradIndices.erase(thrust::remove(gradIndices.begin(), gradIndices.end(), -1), gradIndices.end());
    thrust::sort_by_key(gradIndices.begin(), gradIndices.end(), grad.begin());
    thrust::device_vector<int> outputGradIndices(6 * nConstraints);
    thrust::device_vector<Vector3f> outputGrad(6 * nConstraints);
    auto iter = thrust::reduce_by_key(gradIndices.begin(), gradIndices.end(), grad.begin(), outputGradIndices.begin(), outputGrad.begin());
    addConstraintGradient<<<GRID_SIZE, BLOCK_SIZE>>>(iter.first - outputGradIndices.begin(), pointer(outputGradIndices), pointer(outputGrad), pointer(gradient));
    CUDA_CHECK_LAST();
}