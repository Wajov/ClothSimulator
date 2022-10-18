#include "ImpactZone.hpp"

ImpactZone::ImpactZone() {

}

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
