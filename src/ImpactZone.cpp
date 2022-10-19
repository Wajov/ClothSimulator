#include "ImpactZone.hpp"

ImpactZone::ImpactZone() {}

ImpactZone::~ImpactZone() {}

bool ImpactZone::getActive() const {
    return active;
}

void ImpactZone::setActive(bool active) {
    this->active = active;
}

std::vector<Vertex*>& ImpactZone::getVertices() {
    return vertices;
}

void ImpactZone::addVertex(const Vertex* vertex) {
    vertices.push_back(const_cast<Vertex*>(vertex));
}

std::vector<Impact>& ImpactZone::getImpacts() {
    return impacts;
}

void ImpactZone::addImpact(const Impact& impact) {
    impacts.push_back(impact);
}

bool ImpactZone::contain(const Vertex* vertex) const {
    for (const Vertex* vert : vertices)
        if (vert == vertex)
            return true;
    return false;
}

void ImpactZone::merge(const ImpactZone* zone) {
    vertices.insert(vertices.end(), zone->vertices.begin(), zone->vertices.end());
    impacts.insert(impacts.end(), zone->impacts.begin(), zone->impacts.end());
}
