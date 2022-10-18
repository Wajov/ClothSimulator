#ifndef IMPACT_ZONE_OPTIMIZATION_HPP
#define IMPACT_ZONE_OPTIMIZATION_HPP

#include "Optimization.hpp"
#include "ImpactZone.hpp"

class ImpactZoneOptimization : public Optimization {
private:

public:
    ImpactZoneOptimization(const ImpactZone* zone);
    ~ImpactZoneOptimization();
};

#endif