#ifndef HANDLE_HPP
#define HANDLE_HPP

#include "Vector.cuh"
#include "Vertex.cuh"

class Handle {
private:
    Vertex* vertex;
    Vector3f position;

public:
    Handle(const Vertex* vertex, const Vector3f& position);
    ~Handle();
    Vertex* getVertex() const;
    Vector3f getPosition() const;
};

#endif