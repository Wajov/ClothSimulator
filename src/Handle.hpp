#ifndef HANDLE_HPP
#define HANDLE_HPP

#include "Vector.hpp"
#include "Vertex.hpp"

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