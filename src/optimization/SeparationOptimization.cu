#include "SeparationOptimization.cuh"

SeparationOptimization::SeparationOptimization(const std::vector<Intersection>& intersections, float thickness, int deform, float obstacleArea) :
    intersections(intersections),
    thickness(thickness),
    obstacleArea(obstacleArea) {
    nConstraints = intersections.size();
    std::vector<Pairni> intersectionNodes;
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

SeparationOptimization::~SeparationOptimization() {}

float SeparationOptimization::objective(const std::vector<Vector3f>& x) const {
    float ans = 0.0f;
    for (int i = 0; i < nNodes; i++) {
        Node* node = nodes[i];
        ans += node->area * (x[i] - node->x1).norm2();
    }
    return 0.5f * ans * invArea;
}

float SeparationOptimization::objective(const thrust::device_vector<Vector3f>& x) const {
    // TODO
    return 0.0f;
}

void SeparationOptimization::objectiveGradient(const std::vector<Vector3f>& x, std::vector<Vector3f>& gradient) const {
    for (int i = 0; i < nNodes; i++) {
        Node* node = nodes[i];
        gradient[i] = invArea * node->area * (x[i] - node->x1);
    }
}

void SeparationOptimization::objectiveGradient(const thrust::device_vector<Vector3f>& x, thrust::device_vector<Vector3f>& gradient) const {
    // TODO
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
    // TODO
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
    // TODO
}