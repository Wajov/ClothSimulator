#include "ImpactZone.hpp"

ImpactZone::ImpactZone() {}

ImpactZone::~ImpactZone() {}

bool ImpactZone::getActive() const {
    return active;
}

void ImpactZone::setActive(bool active) {
    this->active = active;
}

std::vector<Node*>& ImpactZone::getNodes() {
    return nodes;
}

void ImpactZone::addNode(const Node* node) {
    nodes.push_back(const_cast<Node*>(node));
}

std::vector<Impact>& ImpactZone::getImpacts() {
    return impacts;
}

void ImpactZone::addImpact(const Impact& impact) {
    impacts.push_back(impact);
}

bool ImpactZone::contain(const Node* node) const {
    for (const Node* n : nodes)
        if (n == node)
            return true;
    return false;
}

void ImpactZone::merge(const ImpactZone* zone) {
    nodes.insert(nodes.end(), zone->nodes.begin(), zone->nodes.end());
    impacts.insert(impacts.end(), zone->impacts.begin(), zone->impacts.end());
}
