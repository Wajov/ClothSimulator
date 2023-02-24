#ifndef DISK_CUH
#define DISK_CUH

#include "MathHelper.cuh"
#include "Vector.cuh"

class Disk {
public:
    Vector2f o;
    float r;
    Disk();
    Disk(const Vector2f& o, float r);
    ~Disk();
    bool enclose(const Disk& d) const;
    static Disk circumscribedDisk(const Disk& d0, const Disk& d1);
    static Disk circumscribedDisk(const Disk& d0, const Disk& d1, const Disk& d2);
};

#endif