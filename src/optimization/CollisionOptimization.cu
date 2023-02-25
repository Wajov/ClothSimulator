#include "CollisionOptimization.cuh"

CollisionOptimization::CollisionOptimization(const std::vector<Impact>& impacts, float thickness, int deform, float obstacleMass) :
    impacts(impacts),
    thickness(thickness),
    obstacleMass(obstacleMass) {
    indices.resize(impacts.size());
    for (int i = 0; i < impacts.size(); i++) {
        indices[i].resize(4);
        const Impact& impact = impacts[i];
        for (int j = 0; j < 4; j++)
            indices[i][j] = addNode(impact.nodes[j], deform);
    }

    nodeSize = nodes.size();
    constraintSize = impacts.size();

    invMass = 0.0f;
    for (const Node* node : nodes) {
        float mass = node->isFree ? node->mass : obstacleMass;
        invMass += 1.0f / mass;
    }
    invMass /= nodes.size();
}

CollisionOptimization::~CollisionOptimization() {}

int CollisionOptimization::addNode(const Node* node, int deform) {
    if (deform == 0 && !node->isFree)
        return -1;
    
    for (int i = 0; i < nodes.size(); i++)
        if (nodes[i] == node)
            return i;

    nodes.push_back(const_cast<Node*>(node));
    return nodes.size() - 1;
}

void CollisionOptimization::initialize(std::vector<Vector3f>& x) const {
    for (int i = 0; i < nodes.size(); i++)
        x[i] = nodes[i]->x;
}

void CollisionOptimization::finalize(const std::vector<Vector3f>& x) {
    for (int i = 0; i < nodes.size(); i++)
        nodes[i]->x = x[i];
}

float CollisionOptimization::objective(const std::vector<Vector3f>& x) const {
    float ans = 0.0f;
    for (int i = 0; i < nodes.size(); i++) {
        Node* node = nodes[i];
        float mass = node->isFree ? node->mass : obstacleMass;
        ans += mass * (x[i] - node->x1).norm2();
    }
    return 0.5f * ans * invMass;
}

void CollisionOptimization::objectiveGradient(const std::vector<Vector3f>& x, std::vector<Vector3f>& gradient) const {
    for (int i = 0; i < nodes.size(); i++) {
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