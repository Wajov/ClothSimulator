#include "Disk.cuh"

Disk::Disk() :
    o(0.0f, 0.0f),
    r(0.0f) {}

Disk::Disk(const Vector2f& o, float r) :
    o(o),
    r(r) {}

Disk::~Disk() {}

bool Disk::enclose(const Disk& d) const {
    return r >= d.r && sqr(r - d.r) >= (o - d.o).norm2();
}
