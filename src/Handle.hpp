#ifndef HANDLE_HPP
#define HANDLE_HPP

#include "TypeHelper.hpp"
#include "Vertex.hpp"
#include "constraint/EqualConstraint.hpp"

class Handle {
private:
    Vertex* vertex;
    Vector3f position;

public:
    Handle(const Vertex* vertex, const Vector3f& position);
    ~Handle();
    Constraint* getConstraint() const;
    Vertex* getVertex() const;
    Vector3f getPosition() const;
};

#endif