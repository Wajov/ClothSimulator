#include "Impact.hpp"

Impact::Impact() {}

Impact::~Impact() {}

bool Impact::operator<(const Impact& impact) const {
    return t < impact.t;
}

bool Impact::contain(const Vertex* vertex) const {
    for (int i = 0; i < 4; i++)
        if (vertices[i] == vertex)
            return true;
    return false;
}

bool Impact::conflict(const Impact& impact) const {
    for (int i = 0; i < 4; i++)
        if (vertices[i]->isFree && impact.contain(vertices[i]))
            return true;
    return false;
}
