#include "SeparationOptimization.hpp"

SeparationOptimization::SeparationOptimization(const std::vector<Intersection>& intersections, float thickness) :
    thickness(thickness),
    intersections(intersections) {
    indices.resize(intersections.size());
    for (int i = 0; i < intersections.size(); i++) {
        indices[i].resize(6);
        const Intersection& intersection = intersections[i];
        for (int j = 0; j < 3; j++)
            indices[i][j] = addNode(intersection.face0->vertices[j]->node);

        for (int j = 0; j < 3; j++)
            indices[i][j + 3] = addNode(intersection.face1->vertices[j]->node);
    }

    nodeSize = nodes.size();
    constraintSize = intersections.size();
    
    invArea = 0.0f;
    for (const Node* node : nodes)
        invArea += 1.0f / node->area;
}

SeparationOptimization::~SeparationOptimization() {}

int SeparationOptimization::addNode(const Node* node) {
    if (!node->isFree)
        return -1;
    
    for (int i = 0; i < nodes.size(); i++)
        if (nodes[i] == node)
            return i;

    nodes.push_back(const_cast<Node*>(node));
    return nodes.size() - 1;
}

void SeparationOptimization::initialize(std::vector<Vector3f>& x) const {
    for (int i = 0; i < nodes.size(); i++)
        x[i] = nodes[i]->x;
}

void SeparationOptimization::finalize(const std::vector<Vector3f>& x) {
    for (int i = 0; i < nodes.size(); i++)
        nodes[i]->x = x[i];
}

float SeparationOptimization::objective(const std::vector<Vector3f>& x) const {
    float ans = 0.0f;
    for (int i = 0; i < nodes.size(); i++) {
        Node* node = nodes[i];
        ans += node->area * (x[i] - node->x1).norm2();
    }
    return 0.5f * ans * invArea;
}

void SeparationOptimization::objectiveGradient(const std::vector<Vector3f>& x, std::vector<Vector3f>& gradient) const {
    for (int i = 0; i < nodes.size(); i++) {
        Node* node = nodes[i];
        gradient[i] = invArea * node->area * (x[i] - node->x1);
    }
}

float SeparationOptimization::constraint(const std::vector<Vector3f>& x, int index, int& sign) const {
    sign = 1;
    float ans = -thickness;
    const Intersection& intersection = intersections[index];
    for (int i = 0; i < 3; i++) {
        int j0 = indices[index][i];
        if (j0 > -1)
            ans += intersection.b0(i) * intersection.d.dot(x[j0]);
        else
            ans += intersection.b0(i) * intersection.d.dot(intersection.face0->vertices[i]->node->x);

        int j1 = indices[index][i + 3];
        if (j1 > -1)
            ans -= intersection.b1(i) * intersection.d.dot(x[j1]);
        else
            ans -= intersection.b1(i) * intersection.d.dot(intersection.face1->vertices[i]->node->x);
    }
    return ans;
}

void SeparationOptimization::constraintGradient(const std::vector<Vector3f>& x, int index, float factor, std::vector<Vector3f>& gradient) const {
    const Intersection& intersection = intersections[index];
    for (int i = 0; i < 3; i++) {
        int j0 = indices[index][i];
        if (j0 > -1)
            gradient[j0] += factor * intersection.b0(i) * intersection.d;
        
        int j1 = indices[index][i + 3];
        if (j1 > -1)
            gradient[j1] -= factor * intersection.b1(i) * intersection.d;
    }
}