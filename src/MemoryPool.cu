#include "MemoryPool.cuh"

MemoryPool::MemoryPool() :
    nodePointer(0),
    vertexPointer(0),
    edgePointer(0),
    facePointer(0) {
    if (!gpu) {
        nodePool.resize(NODE_POOL_SIZE);
        vertexPool.resize(VERTEX_POOL_SIZE);
        edgePool.resize(EDGE_POOL_SIZE);
        facePool.resize(FACE_POOL_SIZE);
    } else {
        nodePoolGpu.resize(NODE_POOL_SIZE);
        vertexPoolGpu.resize(VERTEX_POOL_SIZE);
        edgePoolGpu.resize(EDGE_POOL_SIZE);
        facePoolGpu.resize(FACE_POOL_SIZE);
    }
}

MemoryPool::~MemoryPool() {}

Node* MemoryPool::createNode(const Vector3f& x, bool isFree) {
    if (nodePointer >= NODE_POOL_SIZE) {
        std::cerr << "Memory pool of nodes run out!" << std::endl;
        exit(1);
    }

    Node& node = nodePool[nodePointer++];
    node.x = node.x0 = node.x1 = x;
    node.isFree = isFree;
    node.preserve = false;
    return &node;
}

Vertex* MemoryPool::createVertex(const Vector2f& u) {
    if (vertexPointer >= VERTEX_POOL_SIZE) {
        std::cerr << "Memory pool of vertices run out!" << std::endl;
        exit(1);
    }

    Vertex& vertex = vertexPool[vertexPointer++];
    vertex.u = u;
    return &vertex;
}

Edge* MemoryPool::createEdge(const Node* node0, const Node* node1) {
    if (edgePointer >= EDGE_POOL_SIZE) {
        std::cerr << "Memory pool of edges run out!" << std::endl;
        exit(1);
    }

    Edge& edge = edgePool[edgePointer++];
    edge.nodes[0] = const_cast<Node*>(node0);
    edge.nodes[1] = const_cast<Node*>(node1);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++)
            edge.vertices[i][j] = nullptr;
        edge.opposites[i] = nullptr;
        edge.adjacents[i] = nullptr;
    }
    return &edge;
}

Face* MemoryPool::createFace(const Vertex* vertex0, const Vertex* vertex1, const Vertex* vertex2, const Material* material) {
    if (facePointer >= FACE_POOL_SIZE) {
        std::cerr << "Memory pool of faces run out!" << std::endl;
        exit(1);
    }

    Face& face = facePool[facePointer++];
    face.vertices[0] = const_cast<Vertex*>(vertex0);
    face.vertices[1] = const_cast<Vertex*>(vertex1);
    face.vertices[2] = const_cast<Vertex*>(vertex2);
    face.initialize(material);
    return &face;
}

Node* MemoryPool::createNodes(int nNodes) {
    if (nodePointer + nNodes > NODE_POOL_SIZE) {
        std::cerr << "Memory pool of nodes run out!" << std::endl;
        exit(1);
    }

    nodePointer += nNodes;
    return pointer(nodePoolGpu, nodePointer - nNodes);
}

Vertex* MemoryPool::createVertices(int nVertices) {
    if (vertexPointer + nVertices > VERTEX_POOL_SIZE) {
        std::cerr << "Memory pool of vertices run out!" << std::endl;
        exit(1);
    }

    vertexPointer += nVertices;
    return pointer(vertexPoolGpu, vertexPointer - nVertices);
}

Edge* MemoryPool::createEdges(int nEdges) {
    if (edgePointer + nEdges > EDGE_POOL_SIZE) {
        std::cerr << "Memory pool of edges run out!" << std::endl;
        exit(1);
    }

    edgePointer += nEdges;
    return pointer(edgePoolGpu, edgePointer - nEdges);
}

Face* MemoryPool::createFaces(int nFaces) {
    if (facePointer + nFaces >= FACE_POOL_SIZE) {
        std::cerr << "Memory pool of faces run out!" << std::endl;
        exit(1);
    }

    facePointer += nFaces;
    return pointer(facePoolGpu, facePointer - nFaces);
}