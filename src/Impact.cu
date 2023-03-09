#include "Impact.cuh"

Impact::Impact() :
    t(-1.0f) {}

Impact::~Impact() {}

bool Impact::operator<(const Impact& impact) const {
    return t < impact.t;
}
