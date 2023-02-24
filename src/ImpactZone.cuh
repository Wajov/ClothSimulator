#ifndef IMPACT_ZONE_CUH
#define IMPACT_ZONE_CUH

#include <vector>

#include "Node.cuh"
#include "Impact.cuh"

class ImpactZone {
private:
    bool active;
    std::vector<Node*> nodes;
    std::vector<Impact> impacts;

public:
    ImpactZone();
    ~ImpactZone();
    bool getActive() const;
    void setActive(bool active);
    std::vector<Node*>& getNodes();
    void addNode(const Node* node);
    std::vector<Impact>& getImpacts();
    void addImpact(const Impact& impact);
    bool contain(const Node* node) const;
    void merge(const ImpactZone* zone);
};

#endif