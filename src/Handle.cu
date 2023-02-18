#include "Handle.cuh"

Handle::Handle(const Node* node, const Vector3f& position) :
    node(const_cast<Node*>(node)),
    position(position) {}

Handle::~Handle() {}

Node* Handle::getNode() const {
    return node;
}

Vector3f Handle::getPosition() const {
    return position;
}
