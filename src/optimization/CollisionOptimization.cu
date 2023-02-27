#include "CollisionOptimization.cuh"

CollisionOptimization::CollisionOptimization(const std::vector<Impact>& impacts, float thickness, int deform, float obstacleMass) :
    impacts(impacts),
    thickness(thickness),
    obstacleMass(obstacleMass) {
    nConstraints = impacts.size();
    std::vector<Pair<Node*, Pairii>> nodeImpacts;
    for (int i = 0; i < impacts.size(); i++) {
        const Impact& impact = impacts[i];
        for (int j = 0; j < 4; j++) {
            Node* node = impact.nodes[j];
            if (deform == 1 || node->isFree)
                nodeImpacts.emplace_back(node, Pairii(i, j));
        }
    }

    std::sort(nodeImpacts.begin(), nodeImpacts.end());
    invMass = 0.0f;
    indices.assign(nConstraints, std::vector<int>(4, -1));
    int index = -1;
    for (int i = 0; i < nodeImpacts.size(); i++) {
        Node* node = nodeImpacts[i].first;
        Pairii& idx = nodeImpacts[i].second;
        if (i == 0 || node != nodeImpacts[i - 1].first) {
            float mass = node->isFree ? node->mass : obstacleMass;
            invMass += 1.0f / mass;
            nodes.push_back(node);
            index++;
        }
        indices[idx.first][idx.second] = index;
    }
    nNodes = nodes.size();
    invMass /= nNodes;
}

CollisionOptimization::~CollisionOptimization() {}

void CollisionOptimization::initialize(std::vector<Vector3f>& x) const {
    for (int i = 0; i < nNodes; i++)
        x[i] = nodes[i]->x;
}

void CollisionOptimization::finalize(const std::vector<Vector3f>& x) {
    for (int i = 0; i < nNodes; i++)
        nodes[i]->x = x[i];
}

float CollisionOptimization::objective(const std::vector<Vector3f>& x) const {
    float ans = 0.0f;
    for (int i = 0; i < nNodes; i++) {
        Node* node = nodes[i];
        float mass = node->isFree ? node->mass : obstacleMass;
        ans += mass * (x[i] - node->x1).norm2();
    }
    return 0.5f * ans * invMass;
}

void CollisionOptimization::objectiveGradient(const std::vector<Vector3f>& x, std::vector<Vector3f>& gradient) const {
    for (int i = 0; i < nNodes; i++) {
        Node* node = nodes[i];
        float mass = node->isFree ? node->mass : obstacleMass;
        gradient[i] = invMass * mass * (x[i] - node->x1);
    }
}

float CollisionOptimization::constraint(const std::vector<Vector3f>& x, int index, int& sign) const {
    sign = 1;
    float ans = -thickness;
    const Impact& impact = impacts[index];
    for (int i = 0; i < 4; i++) {
        int j = indices[index][i];
        if (j > -1)
            ans += impact.w[i] * impact.n.dot(x[j]);
        else
            ans += impact.w[i] * impact.n.dot(impact.nodes[i]->x);
    }
    return ans;
}

void CollisionOptimization::constraintGradient(const std::vector<Vector3f>& x, int index, float factor, std::vector<Vector3f>& gradient) const {
    const Impact& impact = impacts[index];
    for (int i = 0; i < 4; i++) {
        int j = indices[index][i];
        if (j > -1)
            gradient[j] += factor * impact.w[i] * impact.n;
    }
}