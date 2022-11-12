#include "Operator.hpp"

Operator::Operator() {}

Operator::~Operator() {}

void Operator::updateActive(const std::vector<Face*>& activeFaces) {
    std::unordered_set<Edge*> edges;
    for (const Face* face : activeFaces)
        for (int i = 0; i < 3; i++)
            edges.insert(face->getEdge(i));
    
    for (Edge* edge : edges)
        activeEdges.push_back(edge);
}

bool Operator::empty() const {
   return addedFaces.empty() && removedFaces.empty();
}

void Operator::flip(const Edge* edge, const Material* material) {
    Vertex* vertex0 = edge->getVertex(0);
    Vertex* vertex1 = edge->getVertex(1);
    Vertex* vertex2 = edge->getOpposite(0);
    Vertex* vertex3 = edge->getOpposite(1);

    Face* face0 = edge->getAdjacent(0);
    Face* face1 = edge->getAdjacent(1);

    Edge* edge0 = face0->findEdge(vertex1, vertex2);
    Edge* edge1 = face0->findEdge(vertex2, vertex0);
    Edge* edge2 = face1->findEdge(vertex0, vertex3);
    Edge* edge3 = face1->findEdge(vertex3, vertex1);

    Edge* newEdge = new Edge(vertex2, vertex3);
    Face* newFace0 = new Face(vertex0, vertex3, vertex2, material);
    Face* newFace1 = new Face(vertex1, vertex2, vertex3, material);
    newEdge->setOppositeAndAdjacent(vertex0, newFace0);
    newEdge->setOppositeAndAdjacent(vertex1, newFace1);
    newFace0->setEdges(edge2, newEdge, edge1);
    newFace1->setEdges(edge0, newEdge, edge3);

    edge0->setOppositeAndAdjacent(vertex3, newFace1);
    edge1->setOppositeAndAdjacent(vertex3, newFace0);
    edge2->setOppositeAndAdjacent(vertex2, newFace0);
    edge3->setOppositeAndAdjacent(vertex2, newFace1);

    addedEdges.push_back(newEdge);
    removedEdges.push_back(const_cast<Edge*>(edge));
    addedFaces.push_back(newFace0);
    addedFaces.push_back(newFace1);
    removedFaces.push_back(face0);
    removedFaces.push_back(face1);

    updateActive(addedFaces);
}

void Operator::split(const Edge* edge, const Material* material) {
    Vertex* vertex0 = edge->getVertex(0);
    Vertex* vertex1 = edge->getVertex(1);
    Vertex* vertex2 = edge->getOpposite(0);
    Vertex* vertex3 = edge->getOpposite(1);
    
    Vertex* newVertex = new Vertex(0.5f * (vertex0->x + vertex1->x), true);
    newVertex->u = 0.5f * (vertex0->u + vertex1->u);
    newVertex->v = 0.5f * (vertex0->v + vertex1->v);
    newVertex->sizing = 0.5f * (vertex0->sizing + vertex1->sizing);
    Edge* newEdge0 = new Edge(newVertex, vertex0);
    Edge* newEdge1 = new Edge(newVertex, vertex1);
    
    addedVertices.push_back(newVertex);
    addedEdges.push_back(newEdge0);
    addedEdges.push_back(newEdge1);
    removedEdges.push_back(const_cast<Edge*>(edge));

    if (vertex2 != nullptr) {
        Face* face0 = edge->getAdjacent(0);
        Edge* edge0 = face0->findEdge(vertex1, vertex2);
        Edge* edge1 = face0->findEdge(vertex2, vertex0);
        
        Edge* newEdge2 = new Edge(newVertex, vertex2);
        Face* newFace0 = new Face(vertex0, newVertex, vertex2, material);
        Face* newFace1 = new Face(vertex2, newVertex, vertex1, material);
        newEdge0->setOppositeAndAdjacent(vertex2, newFace0);
        newEdge1->setOppositeAndAdjacent(vertex2, newFace1);
        newEdge2->setOppositeAndAdjacent(vertex0, newFace0);
        newEdge2->setOppositeAndAdjacent(vertex1, newFace1);
        newFace0->setEdges(newEdge0, newEdge2, edge1);
        newFace1->setEdges(newEdge2, newEdge1, edge0);
        edge0->setOppositeAndAdjacent(newVertex, newFace1);
        edge1->setOppositeAndAdjacent(newVertex, newFace0);
        
        addedEdges.push_back(newEdge2);
        addedFaces.push_back(newFace0);
        addedFaces.push_back(newFace1);
        removedFaces.push_back(face0);
    }

    if (vertex3 != nullptr) {
        Face* face1 = edge->getAdjacent(1);
        Edge* edge2 = face1->findEdge(vertex0, vertex3);
        Edge* edge3 = face1->findEdge(vertex3, vertex1);

        Edge* newEdge3 = new Edge(newVertex, vertex3);
        Face* newFace2 = new Face(vertex0, vertex3, newVertex, material);
        Face* newFace3 = new Face(vertex3, vertex1, newVertex, material);
        newEdge0->setOppositeAndAdjacent(vertex3, newFace2);
        newEdge1->setOppositeAndAdjacent(vertex3, newFace3);
        newEdge3->setOppositeAndAdjacent(vertex0, newFace2);
        newEdge3->setOppositeAndAdjacent(vertex1, newFace3);
        newFace2->setEdges(edge2, newEdge3, newEdge0);
        newFace3->setEdges(edge3, newEdge1, newEdge3);
        edge2->setOppositeAndAdjacent(newVertex, newFace2);
        edge3->setOppositeAndAdjacent(newVertex, newFace3);

        addedEdges.push_back(newEdge3);
        addedFaces.push_back(newFace2);
        addedFaces.push_back(newFace3);
        removedFaces.push_back(face1);
    }

    updateActive(addedFaces);
}

void Operator::collapse(const Edge* edge, bool reverse, const Material* material, std::unordered_map<Vertex*, std::vector<Edge*>>& adjacentEdges, std::unordered_map<Vertex*, std::vector<Face*>>& adjacentFaces) {
    Vertex* vertex0 = edge->getVertex(0);
    Vertex* vertex1 = edge->getVertex(1);
    Vertex* vertex2 = edge->getOpposite(0);
    Vertex* vertex3 = edge->getOpposite(1);

    Face* face0 = edge->getAdjacent(0);
    Face* face1 = edge->getAdjacent(1);
    
    if (reverse) {
        mySwap(vertex0, vertex1);
        mySwap(vertex2, vertex3);
        mySwap(face0, face1);
    }

    removedVertices.push_back(vertex0);
    removedEdges.push_back(const_cast<Edge*>(edge));

    if (vertex2 != nullptr) {
        Edge* edge0 = face0->findEdge(vertex1, vertex2);
        Edge* edge1 = face0->findEdge(vertex2, vertex0);

        if (!edge1->isBoundary()) {
            Vertex* v1 = edge1->getOpposite(0);
            Face* f1 = edge1->getAdjacent(0);
            if (f1 == face0) {
                v1 = edge1->getOpposite(1);
                f1 = edge1->getAdjacent(1);
            }
            
            edge0->replaceOpposite(vertex0, v1);
            edge0->replaceAdjacent(face0, f1);
            f1->replaceEdge(edge1, edge0);
        }

        removedEdges.push_back(edge1);
        removedFaces.push_back(face0);
    }

    if (vertex3 != nullptr) {
        Edge* edge2 = face1->findEdge(vertex0, vertex3);
        Edge* edge3 = face1->findEdge(vertex3, vertex1);

        if (!edge2->isBoundary()) {
            Vertex* v2 = edge2->getOpposite(0);
            Face* f2 = edge2->getAdjacent(0);
            if (f2 == face1) {
                v2 = edge2->getOpposite(1);
                f2 = edge2->getAdjacent(1);
            }

            edge3->replaceOpposite(vertex0, v2);
            edge3->replaceAdjacent(face1, f2);
            f2->replaceEdge(edge2, edge3);
        }

        removedEdges.push_back(edge2);
        removedFaces.push_back(face1);
    }

    std::vector<Edge*>& edges0 = adjacentEdges[vertex0];
    std::vector<Edge*>& edges1 = adjacentEdges[vertex1];
    for (Edge* edge : edges0)
        if (!edge->contain(vertex1)) {
            edge->replaceVertex(vertex0, vertex1);
            edges1.push_back(edge);
        }
    adjacentEdges.erase(vertex0);
    
    std::vector<Face*>& faces0 = adjacentFaces[vertex0];
    std::vector<Face*>& faces1 = adjacentFaces[vertex1];
    std::vector<Face*> activeFaces;
    for (Face* face : faces0)
        if (!face->contain(edge)) {
            face->findOpposite(vertex0)->replaceOpposite(vertex0, vertex1);
            face->replaceVertex(vertex0, vertex1);
            face->initialize(material);
            faces1.push_back(face);
            activeFaces.push_back(face);
        }
    adjacentFaces.erase(vertex0);

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

void Operator::updateAdjacents(std::unordered_map<Vertex*, std::vector<Edge*>>& adjacentEdges, std::unordered_map<Vertex*, std::vector<Face*>>& adjacentFaces) const {
    for (const Edge* edge : removedEdges)
        for (int i = 0; i < 2; i++) {
            std::vector<Edge*>& edges = adjacentEdges[edge->getVertex(i)];
            edges.erase(std::remove(edges.begin(), edges.end(), edge), edges.end());
        }
    for (Edge* edge : addedEdges)
        for (int i = 0; i < 2; i++)
            adjacentEdges[edge->getVertex(i)].push_back(edge);

    for (const Face* face : removedFaces)
        for (int i = 0; i < 3; i++) {
            std::vector<Face*>& faces = adjacentFaces[face->getVertex(i)];
            faces.erase(std::remove(faces.begin(), faces.end(), face), faces.end());
        }
    for (Face* face : addedFaces)
        for (int i = 0; i < 3; i++)
            adjacentFaces[face->getVertex(i)].push_back(face);
}