#include "ImpactZoneOptimization.hpp"

ImpactZoneOptimization::ImpactZoneOptimization(const ImpactZone* zone, double thickness, double obstacleMass) :
    thickness(thickness),
    obstacleMass(obstacleMass) {
    nodes = const_cast<ImpactZone*>(zone)->getNodes();
    impacts = const_cast<ImpactZone*>(zone)->getImpacts();
    variableSize = 3 * nodes.size();
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

    invMass = 0.0;
    for (const Node* node : nodes) {
        double mass = node->isFree ? node->mass : obstacleMass;
        invMass += 1.0 / mass;
    }
    invMass /= nodes.size();
}

ImpactZoneOptimization::~ImpactZoneOptimization() {}

void ImpactZoneOptimization::initialize(double* x) const {
    for (int i = 0; i < nodes.size(); i++)
        for (int j = 0; j < 3; j++)
            x[3 * i + j] = nodes[i]->x(j);
}

void ImpactZoneOptimization::precompute(const double *x) {}

void ImpactZoneOptimization::finalize(const double* x) {
    for (int i = 0; i < nodes.size(); i++)
        for (int j = 0; j < 3; j++)
            nodes[i]->x(j) = static_cast<float>(x[3 * i + j]);
}

double ImpactZoneOptimization::objective(const double* x) const {
    double ans = 0.0;
    for (int i = 0; i < nodes.size(); i++) {
        Node* node = nodes[i];
        double mass = node->isFree ? node->mass : obstacleMass;
        double norm2 = 0.0;
        for (int j = 0; j < 3; j++)
            norm2 += sqr(x[3 * i + j] - node->x1(j));
        ans += 0.5 * mass * norm2;
    }
    return ans * invMass;
}

void ImpactZoneOptimization::objectiveGradient(const double* x, double* gradient) const {
    for (int i = 0; i < nodes.size(); i++) {
        Node* node = nodes[i];
        double mass = node->isFree ? node->mass : obstacleMass;
        for (int j = 0; j < 3; j++)
            gradient[3 * i + j] = invMass * mass * (x[3 * i + j] - node->x1(j));
    }
}

double ImpactZoneOptimization::constraint(const double* x, int index, int& sign) const {
    sign = 1;
    double ans = -thickness;
    const Impact& impact = impacts[index];
    for (int i = 0; i < 4; i++) {
        int j = indices[index][i];
        if (j > -1) {
            double dot = 0.0;
            for (int k = 0; k < 3; k++)
                dot += impact.n(k) * x[3 * j + k];
            ans += impact.w[i] * dot;
        } else
            ans += impact.w[i] * impact.n.dot(impact.nodes[i]->x);
    }
    return ans;
}

void ImpactZoneOptimization::constraintGradient(const double* x, int index, double factor, double* gradient) const {
    const Impact& impact = impacts[index];
    for (int i = 0; i < 4; i++) {
        int j = indices[index][i];
        if (j > -1)
            for (int k = 0; k < 3; k++)
                gradient[3 * j + k] += factor * impact.w[i] * impact.n(k);
    }
}