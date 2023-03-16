#include "Operator.cuh"

Operator::Operator() {}

Operator::~Operator() {}

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
}

void Operator::flip(const thrust::device_vector<Edge*>& edges, const Material* material) {
    int nEdges = edges.size();
    addedEdgesGpu.resize(nEdges);
    removedEdgesGpu.resize(nEdges);
    addedFacesGpu.resize(2 * nEdges);
    removedFacesGpu.resize(2 * nEdges);
    flipGpu<<<GRID_SIZE, BLOCK_SIZE>>>(nEdges, pointer(edges), material, pointer(addedEdgesGpu), pointer(removedEdgesGpu), pointer(addedFacesGpu), pointer(removedFacesGpu));
    CUDA_CHECK_LAST();
}

void Operator::split(const Edge* edge, const Material* material) {
    Node* node0 = edge->nodes[0];
    Node* node1 = edge->nodes[1];

    Node* newNode = new Node(0.5f * (node0->x + node1->x), node0->isFree && node1->isFree);
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
}

void Operator::split(const thrust::device_vector<Edge*>& edges, const Material* material) {
    int nEdges = edges.size();
    addedNodesGpu.resize(nEdges);
    addedVerticesGpu.resize(2 * nEdges);
    addedEdgesGpu.resize(4 * nEdges);
    removedEdgesGpu.resize(nEdges);
    addedFacesGpu.resize(4 * nEdges);
    removedFacesGpu.resize(2 * nEdges);
    splitGpu<<<GRID_SIZE, BLOCK_SIZE>>>(nEdges, pointer(edges), material, pointer(addedNodesGpu), pointer(addedVerticesGpu), pointer(addedEdgesGpu), pointer(removedEdgesGpu), pointer(addedFacesGpu), pointer(removedFacesGpu));
    CUDA_CHECK_LAST();

    addedVerticesGpu.erase(thrust::remove(addedVerticesGpu.begin(), addedVerticesGpu.end(), nullptr), addedVerticesGpu.end());
    addedEdgesGpu.erase(thrust::remove(addedEdgesGpu.begin(), addedEdgesGpu.end(), nullptr), addedEdgesGpu.end());
    addedFacesGpu.erase(thrust::remove(addedFacesGpu.begin(), addedFacesGpu.end(), nullptr), addedFacesGpu.end());
    removedFacesGpu.erase(thrust::remove(removedFacesGpu.begin(), removedFacesGpu.end(), nullptr), removedFacesGpu.end());
}

void Operator::collapse(const Edge* edge, int side, const Material* material, const std::unordered_map<Node*, std::vector<Edge*>>& adjacentEdges, const std::unordered_map<Vertex*, std::vector<Face*>>& adjacentFaces) {
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

            Vertex* v;
            Face* f;
            if (!edge1->isBoundary()) {
                if (edge1->opposites[0] != vertex1) {
                    v = edge1->opposites[0];
                    f = edge1->adjacents[0];
                } else {
                    v = edge1->opposites[1];
                    f = edge1->adjacents[1];
                }

                f->replaceEdge(edge1, edge0);
            } else {
                v = nullptr;
                f = nullptr;
            }
            edge0->replaceOpposite(vertex0, v);
            edge0->replaceAdjacent(face, f);

            removedEdges.push_back(edge1);
            removedFaces.push_back(face);
        }

    const std::vector<Edge*>& edges = adjacentEdges.at(node0);
    for (Edge* adjacentEdge : edges)
        if (adjacentEdge != edge) {
            adjacentEdge->replaceNode(node0, node1);
            adjacentEdge->replaceVertex(edge->vertices[0][side], edge->vertices[0][1 - side]);
            adjacentEdge->replaceVertex(edge->vertices[1][side], edge->vertices[1][1 - side]);
        }

    if (edge->isSeam())
        for (int i = 0; i < 2; i++) {
            Vertex* vertex0 = edge->vertices[i][side];
            Vertex* vertex1 = edge->vertices[i][1 - side];
            removedVertices.push_back(vertex0);

            const std::vector<Face*>& faces = adjacentFaces.at(vertex0);
            for (Face* adjacentFace : faces)
                if (adjacentFace != edge->adjacents[0] && adjacentFace != edge->adjacents[1]) {
                    adjacentFace->findOpposite(vertex0)->replaceOpposite(vertex0, vertex1);
                    adjacentFace->replaceVertex(vertex0, vertex1);
                    adjacentFace->initialize(material);
                }
        }
    else {
        int index = edge->opposites[0] != nullptr ? 0 : 1;
        Vertex* vertex0 = edge->vertices[index][side];
        Vertex* vertex1 = edge->vertices[index][1 - side];
        removedVertices.push_back(vertex0);
        
        const std::vector<Face*>& faces = adjacentFaces.at(vertex0);
        for (Face* adjacentFace : faces)
            if (adjacentFace != edge->adjacents[0] && adjacentFace != edge->adjacents[1]) {
                adjacentFace->findOpposite(vertex0)->replaceOpposite(vertex0, vertex1);
                adjacentFace->replaceVertex(vertex0, vertex1);
                adjacentFace->initialize(material);
            }
    }
}

void Operator::collapse(const thrust::device_vector<PairEi>& edges, const Material* material, const thrust::device_vector<int>& edgeBegin, const thrust::device_vector<int>& edgeEnd, const thrust::device_vector<Edge*>& adjacentEdges, const thrust::device_vector<int>& faceBegin, const thrust::device_vector<int>& faceEnd, const thrust::device_vector<Face*>& adjacentFaces) {
    int nEdges = edges.size();
    removedNodesGpu.resize(nEdges);
    removedVerticesGpu.resize(2 * nEdges);
    removedEdgesGpu.resize(3 * nEdges);
    removedFacesGpu.resize(2 * nEdges);
    collapseGpu<<<GRID_SIZE, BLOCK_SIZE>>>(nEdges, pointer(edges), material, pointer(edgeBegin), pointer(edgeEnd), pointer(adjacentEdges), pointer(faceBegin), pointer(faceEnd), pointer(adjacentFaces), pointer(removedNodesGpu), pointer(removedVerticesGpu), pointer(removedEdgesGpu), pointer(removedFacesGpu));
    CUDA_CHECK_LAST();

    removedVerticesGpu.erase(thrust::remove(removedVerticesGpu.begin(), removedVerticesGpu.end(), nullptr), removedVerticesGpu.end());
    removedEdgesGpu.erase(thrust::remove(removedEdgesGpu.begin(), removedEdgesGpu.end(), nullptr), removedEdgesGpu.end());
    removedFacesGpu.erase(thrust::remove(removedFacesGpu.begin(), removedFacesGpu.end(), nullptr), removedFacesGpu.end());
}
