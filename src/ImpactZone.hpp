#ifndef IMPACT_ZONE_HPP
#define IMPACT_ZONE_HPP

#include <vector>

#include "Vertex.hpp"

class ImpactZone {
private:
    bool active;
    std::vector<Vertex*> vertices;

public:
    ImpactZone();
    ~ImpactZone();
    bool getActive() const;
    void setActive(bool active);
    std::vector<Vertex*>& getVertices();
};

#endif