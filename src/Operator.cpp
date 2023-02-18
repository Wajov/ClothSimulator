#include "Operator.hpp"

Operator::Operator() {}

Operator::~Operator() {}

void Operator::updateActive(const std::vector<Face*>& activeFaces) {
    std::vector<Edge*> edges(3 * activeFaces.size());
    for (int i = 0; i < activeFaces.size(); i++) {
        Face* face = activeFaces[i];
        for (int j = 0; j < 3; j++)


            edges[3 * i + j] = face->edges[j];
    }
    
    std::sort(edges.begin(), edges.end(), [](const Edge* edge0, const Edge* edge1) {
        return edge0->nodes[0]->index < edge1->nodes[0]->index || edge0->nodes[0]->index == edge1->nodes[0]->index && edge0->nodes[1]->index < edge1->nodes[1]->index;
    });
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    activeEdges.insert(activeEdges.end(), edges.begin(), edges.end());
}

bool Operator::empty() const {
   return addedFaces.empty() && removedFaces.empty();
}

void Operator::flip(const Edge* edge, const Material* material) {
    Vertex* vertex0 = edge->vertices[0][0];
    Vertex* vertex1 = edge->vertices[1][1];
    Vertex* vertex2 = edge->opposites[0];
    Vertex* vertex3 = edge->opposites[1];

    Face* face0 = edge->adjacents[0];
    Face* face1 = edge->adjacents[1];

    Edge* edge0 = face0->findEdge(vertex1, vertex2);
    Edge* edge1 = face0->findEdge(vertex2, vertex0);
    Edge* edge2 = face1->findEdge(vertex0, vertex3);
    Edge* edge3 = face1->findEdge(vertex3, vertex1);

    Edge* newEdge = new Edge(vertex2->node, vertex3->node);
    Face* newFace0 = new Face(vertex0, vertex3, vertex2, material);
    Face* newFace1 = new Face(vertex1, vertex2, vertex3, material);
    newEdge->initialize(vertex0, newFace0);
    newEdge->initialize(vertex1, newFace1);
    newFace0->setEdges(edge2, newEdge, edge1);
    newFace1->setEdges(edge0, newEdge, edge3);

    edge0->initialize(vertex3, newFace1);
    edge1->initialize(vertex3, newFace0);
    edge2->initialize(vertex2, newFace0);
    edge3->initialize(vertex2, newFace1);

    addedEdges.push_back(newEdge);
    removedEdges.push_back(const_cast<Edge*>(edge));
    addedFaces.push_back(newFace0);
    addedFaces.push_back(newFace1);
    removedFaces.push_back(face0);
    removedFaces.push_back(face1);

    updateActive(addedFaces);
}

void Operator::split(const Edge* edge, const Material* material, int index) {
    Node* node0 = edge->nodes[0];
    Node* node1 = edge->nodes[1];

    Node* newNode = new Node(0.5f * (node0->x + node1->x), node0->isFree && node1->isFree);
    newNode->index = index;
    newNode->x0 = 0.5f * (node0->x0 + node1->x0);
    newNode->v = 0.5f * (node0->v + node1->v);
    Edge* newEdges[2];
    newEdges[0] = new Edge(newNode, node0);
    newEdges[1] = new Edge(newNode, node1);
    
    addedNodes.push_back(newNode);
    addedEdges.push_back(newEdges[0]);
    addedEdges.push_back(newEdges[1]);
    removedEdges.push_back(const_cast<Edge*>(edge));

    Vertex* newVertices[2];
    if (edge->isSeam()) {
        newVertices[0] = new Vertex(0.5f * (edge->vertices[0][0]->u + edge->vertices[0][1]->u));
        newVertices[0]->sizing = 0.5f * (edge->vertices[0][0]->sizing + edge->vertices[0][1]->sizing);
        newVertices[1] = new Vertex(0.5f * (edge->vertices[1][0]->u + edge->vertices[1][1]->u));
        newVertices[1]->sizing = 0.5f * (edge->vertices[1][0]->sizing + edge->vertices[1][1]->sizing);
        newVertices[0]->node = newVertices[1]->node = newNode;
        addedVertices.push_back(newVertices[0]);
        addedVertices.push_back(newVertices[1]);
    } else {
        int i = edge->opposites[0] != nullptr ? 0 : 1;
        newVertices[0] = newVertices[1] = new Vertex(0.5f * (edge->vertices[i][0]->u + edge->vertices[i][1]->u));
        newVertices[0]->sizing = 0.5f * (edge->vertices[i][0]->sizing + edge->vertices[i][1]->sizing);
        newVertices[0]->node = newNode;
        addedVertices.push_back(newVertices[0]);
    }

    for (int i = 0; i < 2; i++)
        if (edge->opposites[i] != nullptr) {
            Vertex* vertex0 = edge->vertices[i][i];
            Vertex* vertex1 = edge->vertices[i][1 - i];
            Vertex* vertex2 = edge->opposites[i];

            Face* face = edge->adjacents[i];
            Edge* edge0 = face->findEdge(vertex1, vertex2);
            Edge* edge1 = face->findEdge(vertex2, vertex0);
            
            Vertex* newVertex = newVertices[i];
            Edge* newEdge0 = newEdges[i];
            Edge* newEdge1 = newEdges[1 - i];
            Edge* newEdge2 = new Edge(newNode, vertex2->node);
            Face* newFace0 = new Face(vertex0, newVertex, vertex2, material);
            Face* newFace1 = new Face(vertex2, newVertex, vertex1, material);
            
            newEdge0->initialize(vertex2, newFace0);
            newEdge1->initialize(vertex2, newFace1);
            newEdge2->initialize(vertex0, newFace0);
            newEdge2->initialize(vertex1, newFace1);
            newFace0->setEdges(newEdge0, newEdge2, edge1);
            newFace1->setEdges(newEdge2, newEdge1, edge0);
            edge0->initialize(newVertex, newFace1);
            edge1->initialize(newVertex, newFace0);
            
            addedEdges.push_back(newEdge2);
            addedFaces.push_back(newFace0);
            addedFaces.push_back(newFace1);
            removedFaces.push_back(face);
        }

    updateActive(addedFaces);
}

void Operator::collapse(const Edge* edge, int side, const Material* material, std::unordered_map<Node*, std::vector<Edge*>>& adjacentEdges, std::unordered_map<Vertex*, std::vector<Face*>>& adjacentFaces) {
    Node* node0 = edge->nodes[side];
    Node* node1 = edge->nodes[1 - side];

    removedNodes.push_back(node0);
    removedEdges.push_back(const_cast<Edge*>(edge));

    for (int i = 0; i < 2; i++)
        if (edge->opposites[i] != nullptr) {
            Vertex* vertex0 = edge->vertices[i][side];
            Vertex* vertex1 = edge->vertices[i][1 - side];
            Vertex* vertex2 = edge->opposites[i];

            Face* face = edge->adjacents[i];
            Edge* edge0 = face->findEdge(vertex1, vertex2);
            Edge* edge1 = face->findEdge(vertex2, vertex0);

            if (!edge1->isBoundary()) {
                Vertex* v = edge1->opposites[0];
                Face* f = edge1->adjacents[0];
                if (f == face) {
                    v = edge1->opposites[1];
                    f = edge1->adjacents[1];
                }

                edge0->replaceOpposite(vertex0, v);
                edge0->replaceAdjacent(face, f);
                f->replaceEdge(edge1, edge0);
            }

            removedEdges.push_back(edge1);
            removedFaces.push_back(face);
        }

    std::vector<Edge*>& adjacentEdges0 = adjacentEdges[node0];
    std::vector<Edge*>& adjacentEdges1 = adjacentEdges[node1];
    for (Edge* adjacentEdge : adjacentEdges0)
        if (adjacentEdge != edge) {
            adjacentEdge->replaceNode(node0, node1);
            adjacentEdge->replaceVertex(edge->vertices[0][side], edge->vertices[0][1 - side]);
            adjacentEdge->replaceVertex(edge->vertices[1][side], edge->vertices[1][1 - side]);
            adjacentEdges1.push_back(adjacentEdge);
        }
    adjacentEdges.erase(node0);

    std::vector<Face*> activeFaces;
    if (edge->isSeam())
        for (int i = 0; i < 2; i++) {
            Vertex* vertex0 = edge->vertices[i][side];
            Vertex* vertex1 = edge->vertices[i][1 - side];
            removedVertices.push_back(vertex0);

            std::vector<Face*>& adjacentFaces0 = adjacentFaces[vertex0];
            std::vector<Face*>& adjacentFaces1 = adjacentFaces[vertex1];
            for (Face* adjacentFace : adjacentFaces0)
                if (adjacentFace != edge->adjacents[0] && adjacentFace != edge->adjacents[1]) {
                    adjacentFace->findOpposite(vertex0)->replaceOpposite(vertex0, vertex1);
                    adjacentFace->replaceVertex(vertex0, vertex1);
                    adjacentFace->initialize(material);
                    adjacentFaces1.push_back(adjacentFace);
                    activeFaces.push_back(adjacentFace);
                }
            adjacentFaces.erase(vertex0);
        }
    else {
        int index = edge->opposites[0] != nullptr ? 0 : 1;
        Vertex* vertex0 = edge->vertices[index][side];
        Vertex* vertex1 = edge->vertices[index][1 - side];
        removedVertices.push_back(vertex0);
        
        std::vector<Face*>& adjacentFaces0 = adjacentFaces[vertex0];
        std::vector<Face*>& adjacentFaces1 = adjacentFaces[vertex1];
        for (Face* adjacentFace : adjacentFaces0)
            if (adjacentFace != edge->adjacents[0] && adjacentFace != edge->adjacents[1]) {
                adjacentFace->findOpposite(vertex0)->replaceOpposite(vertex0, vertex1);
                adjacentFace->replaceVertex(vertex0, vertex1);
                adjacentFace->initialize(material);
                adjacentFaces1.push_back(adjacentFace);
                activeFaces.push_back(adjacentFace);
            }
        adjacentFaces.erase(vertex0);
    }

    updateActive(activeFaces);
}

void Operator::update(std::vector<Edge*>& edges) const {
    for (const Edge* edge : removedEdges)
        edges.erase(std::remove(edges.begin(), edges.end(), edge), edges.end());
    edges.insert(edges.end(), addedEdges.begin(), addedEdges.end());
}

void Operator::setNull(std::vector<Edge*>& edges) const {
    for (const Edge* edge : removedEdges) {
        std::vector<Edge*>::iterator iter = std::find(edges.begin(), edges.end(), edge);
        if (iter != edges.end())
            *iter = nullptr;
    }
}

void Operator::updateAdjacents(std::unordered_map<Node*, std::vector<Edge*>>& adjacentEdges, std::unordered_map<Vertex*, std::vector<Face*>>& adjacentFaces) const {
    for (const Edge* edge : removedEdges)
        for (int i = 0; i < 2; i++) {
            std::vector<Edge*>& edges = adjacentEdges[edge->nodes[i]];
            edges.erase(std::remove(edges.begin(), edges.end(), edge), edges.end());
        }
    for (Edge* edge : addedEdges)
        for (int i = 0; i < 2; i++)
            adjacentEdges[edge->nodes[i]].push_back(edge);

    for (const Face* face : removedFaces)
        for (int i = 0; i < 3; i++) {
            std::vector<Face*>& faces = adjacentFaces[face->vertices[i]];
            faces.erase(std::remove(faces.begin(), faces.end(), face), faces.end());
        }
    for (Face* face : addedFaces)
        for (int i = 0; i < 3; i++)
            adjacentFaces[face->vertices[i]].push_back(face);
}