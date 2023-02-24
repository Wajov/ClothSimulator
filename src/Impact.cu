#include "Impact.cuh"

Impact::Impact() {}

Impact::~Impact() {}

bool Impact::operator<(const Impact& impact) const {
    return t < impact.t;
}
