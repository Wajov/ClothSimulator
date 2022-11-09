#ifndef IMPACT_ZONE_HPP
#define IMPACT_ZONE_HPP

#include <vector>

#include "Vertex.cuh"
#include "Impact.hpp"

class ImpactZone {
private:
    bool active;
    std::vector<Vertex*> vertices;
    std::vector<Impact> impacts;

public:
    ImpactZone();
    ~ImpactZone();
    bool getActive() const;
    void setActive(bool active);
    std::vector<Vertex*>& getVertices();
    void addVertex(const Vertex* vertex);
    std::vector<Impact>& getImpacts();
    void addImpact(const Impact& impact);
    bool contain(const Vertex* vertex) const;
    void merge(const ImpactZone* zone);
};

#endif