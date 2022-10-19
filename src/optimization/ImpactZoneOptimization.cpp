#include "ImpactZoneOptimization.hpp"

ImpactZoneOptimization::ImpactZoneOptimization(const ImpactZone* zone, double thickness, double obstacleMass) :
    thickness(thickness),
    obstacleMass(obstacleMass) {
    vertices = const_cast<ImpactZone*>(zone)->getVertices();
    impacts = const_cast<ImpactZone*>(zone)->getImpacts();
    variableSize = 3 * vertices.size();
    constraintSize = impacts.size();

    indices.resize(impacts.size());
    for (int i = 0; i < impacts.size(); i++) {
        indices[i].resize(4);
        for (int j = 0; j < 4; j++) {
            indices[i][j] = -1;
            for (int k = 0; k < vertices.size(); k++)
                if (vertices[k] == impacts[i].vertices[j]) {
                    indices[i][j] = k;
                    break;
                }
        }
    }

    for (const Vertex* vertex : vertices) {
        double mass = vertex->isFree ? vertex->m : obstacleMass;
        invMass += 1.0 / mass;
    }
    invMass /= vertices.size();
}

ImpactZoneOptimization::~ImpactZoneOptimization() {}

void ImpactZoneOptimization::initialize(double* x) const {
    for (int i = 0; i < vertices.size(); i++)
        for (int j = 0; j < 3; j++)
            x[3 * i + j] = vertices[i]->x(j);
}

void ImpactZoneOptimization::precompute(const double *x) {}

void ImpactZoneOptimization::finalize(const double* x) {
    for (int i = 0; i < vertices.size(); i++)
        for (int j = 0; j < 3; j++)
            vertices[i]->x(j) = x[3 * i + j];
}

double ImpactZoneOptimization::objective(const double* x) const {
    double ans = 0.0;
    for (int i = 0; i < vertices.size(); i++) {
        Vertex* vertex = vertices[i];
        double mass = vertex->isFree ? vertex->m : obstacleMass;
        double norm2 = 0.0;
        for (int j = 0; j < 3; j++)
            norm2 += sqr(x[3 * i + j] - vertex->x(j));
        ans += 0.5 * mass * norm2;
    }
    return ans * invMass;
}

void ImpactZoneOptimization::objectiveGradient(const double* x, double* gradient) const {
    for (int i = 0; i < vertices.size(); i++) {
        Vertex* vertex = vertices[i];
        double mass = vertex->isFree ? vertex->m : obstacleMass;
        for (int j = 0; j < 3; j++)
            gradient[3 * i + j] = invMass * mass * (x[3 * i + j] - vertex->x(j));
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
            ans += impact.w[i] * impact.n.dot(impact.vertices[i]->x);
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