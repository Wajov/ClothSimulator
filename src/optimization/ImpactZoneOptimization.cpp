#include "ImpactZoneOptimization.hpp"

ImpactZoneOptimization::ImpactZoneOptimization(const ImpactZone* zone, float thickness, float obstacleMass) :
    thickness(thickness),
    obstacleMass(obstacleMass) {
    nodes = const_cast<ImpactZone*>(zone)->getNodes();
    impacts = const_cast<ImpactZone*>(zone)->getImpacts();
    nodeSize = nodes.size();
    constraintSize = impacts.size();

    indices.resize(impacts.size());
    for (int i = 0; i < impacts.size(); i++) {
        indices[i].resize(4);
        for (int j = 0; j < 4; j++) {
            indices[i][j] = -1;
            for (int k = 0; k < nodes.size(); k++)
                if (nodes[k] == impacts[i].nodes[j]) {
                    indices[i][j] = k;
                    break;
                }
        }
    }

    invMass = 0.0f;
    for (const Node* node : nodes) {
        float mass = node->isFree ? node->mass : obstacleMass;
        invMass += 1.0f / mass;
    }
    invMass /= nodes.size();
}

ImpactZoneOptimization::~ImpactZoneOptimization() {}

void ImpactZoneOptimization::initialize(std::vector<Vector3f>& x) const {
    for (int i = 0; i < nodes.size(); i++)
        x[i] = nodes[i]->x;
}

void ImpactZoneOptimization::finalize(const std::vector<Vector3f>& x) {
    for (int i = 0; i < nodes.size(); i++)
        nodes[i]->x = x[i];
}

float ImpactZoneOptimization::objective(const std::vector<Vector3f>& x) const {
    float ans = 0.0f;
    for (int i = 0; i < nodes.size(); i++) {
        Node* node = nodes[i];
        float mass = node->isFree ? node->mass : obstacleMass;
        ans += mass * (x[i] - node->x1).norm2();
    }
    return 0.5f * ans * invMass;
}

void ImpactZoneOptimization::objectiveGradient(const std::vector<Vector3f>& x, std::vector<Vector3f>& gradient) const {
    for (int i = 0; i < nodes.size(); i++) {
        Node* node = nodes[i];
        float mass = node->isFree ? node->mass : obstacleMass;
        gradient[i] = invMass * mass * (x[i] - node->x1);
    }
}

float ImpactZoneOptimization::constraint(const std::vector<Vector3f>& x, int index, int& sign) const {
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

void ImpactZoneOptimization::constraintGradient(const std::vector<Vector3f>& x, int index, float factor, std::vector<Vector3f>& gradient) const {
    const Impact& impact = impacts[index];
    for (int i = 0; i < 4; i++) {
        int j = indices[index][i];
        if (j > -1)
            gradient[j] += factor * impact.w[i] * impact.n;
    }
}