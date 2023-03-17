#include "Edge.cuh"

Edge::Edge(const Node* node0, const Node* node1) :
    nodes{const_cast<Node*>(node0), const_cast<Node*>(node1)},
    vertices{nullptr, nullptr, nullptr, nullptr},
    opposites{nullptr, nullptr},
    adjacents{nullptr, nullptr} {}

Edge::~Edge() {}

void Edge::initialize(const Vertex* vertex, const Face* face) {
    for (int i = 0; i < 3; i++) {
        Vertex* vertex0 = face->vertices[i];
        Vertex* vertex1 = face->vertices[(i + 1) % 3];
        if (nodes[0] == vertex0->node && nodes[1] == vertex1->node) {
            vertices[0][0] = vertex0;
            vertices[0][1] = vertex1;
            opposites[0] = const_cast<Vertex*>(vertex);
            adjacents[0] = const_cast<Face*>(face);
            return;
        } else if (nodes[1] == vertex0->node && nodes[0] == vertex1->node) {
            vertices[1][0] = vertex1;
            vertices[1][1] = vertex0;
            opposites[1] = const_cast<Vertex*>(vertex);
            adjacents[1] = const_cast<Face*>(face);
            return;
        }
    }
}

void Edge::replaceNode(const Node* n, const Node* node) {
    for (int i = 0; i < 2; i++)
        if (nodes[i] == n) {
            nodes[i] = const_cast<Node*>(node);
            return;
        }
}

void Edge::replaceVertex(const Vertex* v, const Vertex* vertex) {
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            if (vertices[i][j] == v)
                vertices[i][j] = const_cast<Vertex*>(vertex);
}

void Edge::replaceOpposite(const Vertex* v, const Vertex* vertex) {
    for (int i = 0; i < 2; i++)
        if (opposites[i] == v) {
            opposites[i] = const_cast<Vertex*>(vertex);
            return;
        }
}

void Edge::replaceAdjacent(const Face* f, const Face* face) {
    for (int i = 0; i < 2; i++)
        if (adjacents[i] == f) {
            adjacents[i] = const_cast<Face*>(face);
            return;
        }
}

bool Edge::isFree() const {
    return nodes[0]->isFree && nodes[1]->isFree;
}

bool Edge::isBoundary() const {
    return adjacents[0] == nullptr || adjacents[1] == nullptr;
}

bool Edge::isSeam() const {
    return adjacents[0] != nullptr && adjacents[1] != nullptr && (vertices[0][0] != vertices[1][0] || vertices[0][1] != vertices[1][1]);
}

Bounds Edge::bounds(bool ccd) const {
    Bounds ans;
    for (int i = 0; i < 2; i++) {
        const Node* node = nodes[i];
        ans += node->x;
        if (ccd)
            ans += node->x0;
    }
    return ans;
}

float Edge::area() const {
    float ans = 0.0f;
    for (int i = 0; i < 2; i++)
        if (adjacents[i] != nullptr)
            ans += adjacents[i]->area;
    return ans;
}

float Edge::length() const {
    return (nodes[1]->x - nodes[0]->x).norm();
}

float Edge::angle() const {
    if (!isBoundary()) {
        Vector3f e = (nodes[0]->x - nodes[1]->x).normalized();
        Vector3f n0 = adjacents[0]->n;
        Vector3f n1 = adjacents[1]->n;
        float sine = e.dot(n0.cross(n1));
        float cosine = n0.dot(n1);
        return atan2(sine, cosine);
    } else
        return 0.0f;
}
