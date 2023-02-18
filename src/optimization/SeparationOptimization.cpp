#include "SeparationOptimization.hpp"

SeparationOptimization::SeparationOptimization(const std::vector<Intersection>& intersections, double thickness) :
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

    variableSize = 3 * nodes.size();
    constraintSize = intersections.size();
    
    invArea = 0.0;
    for (const Node* node : nodes)
        invArea += 1.0 / node->area;
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

void SeparationOptimization::initialize(double* x) const {
    for (int i = 0; i < nodes.size(); i++)
        for (int j = 0; j < 3; j++)
            x[3 * i + j] = nodes[i]->x(j);
}

void SeparationOptimization::precompute(const double *x) {}

void SeparationOptimization::finalize(const double* x) {
    for (int i = 0; i < nodes.size(); i++)
        for (int j = 0; j < 3; j++)
            nodes[i]->x(j) = static_cast<float>(x[3 * i + j]);
}

double SeparationOptimization::objective(const double* x) const {
    double ans = 0.0;
    for (int i = 0; i < nodes.size(); i++) {
        Node* node = nodes[i];
        double norm2 = 0.0;
        for (int j = 0; j < 3; j++)
            norm2 += sqr(x[3 * i + j] - node->x1(j));
        ans += 0.5 * node->area * norm2;
    }
    return ans * invArea;
}

void SeparationOptimization::objectiveGradient(const double* x, double* gradient) const {
    for (int i = 0; i < nodes.size(); i++) {
        Node* node = nodes[i];
        for (int j = 0; j < 3; j++)
            gradient[3 * i + j] = invArea * node->area * (x[3 * i + j] - node->x1(j));
    }
}

double SeparationOptimization::constraint(const double* x, int index, int& sign) const {
    sign = 1;
    double ans = -thickness;
    const Intersection& intersection = intersections[index];
    for (int i = 0; i < 3; i++) {
        int j0 = indices[index][i];
        if (j0 > -1) {
            double dot = 0.0;
            for (int k = 0; k < 3; k++)
                dot += intersection.d(k) * x[3 * j0 + k];
            ans += intersection.b0(i) * dot;
        } else
            ans += intersection.b0(i) * intersection.d.dot(intersection.face0->vertices[i]->node->x);

        int j1 = indices[index][i + 3];
        if (j1 > -1) {
            double dot = 0.0;
            for (int k = 0; k < 3; k++)
                dot += intersection.d(k) * x[3 * j1 + k];
            ans -= intersection.b1(i) * dot;
        } else
            ans -= intersection.b1(i) * intersection.d.dot(intersection.face1->vertices[i]->node->x);
    }
    return ans;
}

void SeparationOptimization::constraintGradient(const double* x, int index, double factor, double* gradient) const {
    const Intersection& intersection = intersections[index];
    for (int i = 0; i < 3; i++) {
        int j0 = indices[index][i];
        if (j0 > -1)
            for (int k = 0; k < 3; k++)
                gradient[3 * j0 + k] += factor * intersection.b0(i) * intersection.d(k);
        
        int j1 = indices[index][i + 3];
        if (j1 > -1)
            for (int k = 0; k < 3; k++)
                gradient[3 * j1 + k] -= factor * intersection.b1(i) * intersection.d(k);
    }
}