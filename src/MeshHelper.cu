#include "MeshHelper.cuh"

__global__ void initializeNodes(int nNodes, const Vector3f* x, bool isFree, Node** nodes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads)
        nodes[i] = new Node(x[i], isFree);
}

__global__ void initializeVertices(int nVertices, const Vector2f* u, Vertex** vertices) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nVertices; i += nThreads)
        vertices[i] = new Vertex(u[i]);
}

__device__ void setEdgeData(int index0, int index1, const Vertex* vertex, const Face* face, Pairii& index, EdgeData& edgeData) {
    if (index0 > index1)
        mySwap(index0, index1);
    
    index.first = index0;
    index.second = index1;
    edgeData.opposite = const_cast<Vertex*>(vertex);
    edgeData.adjacent = const_cast<Face*>(face);
}

__global__ void initializeFaces(int nFaces, const int* xIndices, const int* uIndices, const Node* const* nodes, const Material* material, Vertex** vertices, Face** faces, Pairii* edgeIndices, EdgeData* edgeData) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads) {
        int index0 = 3 * i;
        int index1 = 3 * i + 1;
        int index2 = 3 * i + 2;
        int xIndex0 = xIndices[index0];
        int xIndex1 = xIndices[index1];
        int xIndex2 = xIndices[index2];
        int uIndex0 = uIndices[index0];
        int uIndex1 = uIndices[index1];
        int uIndex2 = uIndices[index2];
        const Node* node0 = nodes[xIndex0];
        const Node* node1 = nodes[xIndex1];
        const Node* node2 = nodes[xIndex2];
        Vertex* vertex0 = vertices[uIndex0];
        Vertex* vertex1 = vertices[uIndex1];
        Vertex* vertex2 = vertices[uIndex2];
        vertex0->node = const_cast<Node*>(node0);
        vertex1->node = const_cast<Node*>(node1);
        vertex2->node = const_cast<Node*>(node2);
        Face* face = new Face(vertex0, vertex1, vertex2, material);
        setEdgeData(xIndex0, xIndex1, vertex2, face, edgeIndices[index0], edgeData[index0]);
        setEdgeData(xIndex1, xIndex2, vertex0, face, edgeIndices[index1], edgeData[index1]);
        setEdgeData(xIndex2, xIndex0, vertex1, face, edgeIndices[index2], edgeData[index2]);
        faces[i] = face;
    }
}

__global__ void initializeEdges(int nEdges, const Pairii* indices, const EdgeData* edgeData, const Node* const* nodes, Edge** edges) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads)
        if (i == 0 || indices[i] != indices[i - 1]) {
            const Pairii& index = indices[i];
            const EdgeData& e = edgeData[i];
            Edge* edge = new Edge(nodes[index.first], nodes[index.second]);
            edge->initialize(e.opposite, e.adjacent);
            e.adjacent->setEdge(edge);
            edges[i] = edge;
        } else
            edges[i] = nullptr;
}

__global__ void setEdges(int nEdges, const Pairii* indices, const EdgeData* edgeData, Edge** edges) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads)
        if (i > 0 && indices[i] == indices[i - 1]) {
            const EdgeData& e = edgeData[i];
            Edge* edge = edges[i - 1];
            edge->initialize(e.opposite, e.adjacent);
            e.adjacent->setEdge(edge);
        }
}

__global__ void setPreserve(int nEdges, const Edge* const* edges) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads) {
        const Edge* edge = edges[i];
        if (edge->isBoundary() || edge->isSeam())
            for (int j = 0; j < 2; j++)
                edge->nodes[j]->preserve = true;
    }
}

__global__ void resetGpu(int nNodes, Node** nodes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads) {
        Node* node = nodes[i];
        node->x = node->x0;
    }
}

__global__ void setBackupFaces(int nFaces, const Face* const* faces, BackupFace* backupFaces) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads) {
        BackupFace& backupFace = backupFaces[i];
        const Face* face = faces[i];
        for (int j = 0; j < 3; j++) {
            Vertex* vertex = face->vertices[j];
            backupFace.x[j] = vertex->node->x;
            backupFace.u[j] = vertex->u;
        }
    }
}

__global__ void initializeIndices(int n, int* indices) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += nThreads)
        indices[i] = i;
}

__global__ void initializeNodeStructures(int nNodes, Node** nodes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads) {
        Node* node = nodes[i];
        node->index = i;
        node->mass = 0.0f;
        node->area = 0.0f;
    }
}

__global__ void initializeVertexStructures(int nVertices, Vertex** vertices) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nVertices; i += nThreads)
        vertices[i]->index = i;
}

__global__ void updateStructuresGpu(int nFaces, const Face* const* faces) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads) {
        const Face* face = faces[i];
        float mass = face->mass / 3.0f;
        float area = face->area;
        for (int j = 0; j < 3; j++) {
            Node* node = face->vertices[j]->node;
            atomicAdd(&node->mass, mass);
            atomicAdd(&node->area, area);
        }
    }
}

__global__ void initializeNodeGeometries(int nNodes, Node** nodes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads) {
        Node* node = nodes[i];
        node->x1 = node->x;
        node->n = Vector3f();
    }
}

__global__ void updateNodeGeometriesGpu(int nFaces, const Face* const* faces) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads) {
        const Face* face = faces[i];
        for (int j = 0; j < 3; j++) {
            Node* node = face->vertices[j]->node;
            Vector3f e0 = face->vertices[(j + 1) % 3]->node->x - node->x;
            Vector3f e1 = face->vertices[(j + 2) % 3]->node->x - node->x;
            Vector3f n = e0.cross(e1) / (e0.norm2() * e1.norm2());
            for (int k = 0; k < 3; k++)
                atomicAdd(&node->n(k), n(k));
        }
    }
}

__global__ void finalizeNodeGeometries(int nNodes, Node** nodes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads)
        nodes[i]->n.normalize();
}

__global__ void updateFaceGeometriesGpu(int nFaces, Face** faces) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads)
        faces[i]->update();
}

__global__ void updateVelocitiesGpu(int nNodes, float invDt, Node** nodes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads) {
        Node* node = nodes[i];
        node->v = (node->x - node->x0) * invDt;
    }
}

__global__ void updateRenderingDataGpu(int nFaces, const Face* const* faces, Renderable* renderables) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads) {
        const Face* face = faces[i];
        for (int j = 0; j < 3; j++) {
            Vertex* vertex = face->vertices[j];
            Node* node = vertex->node;
            int index = 3 * i + j;
            renderables[index].x = node->x;
            renderables[index].n = node->n;
            renderables[index].u = vertex->u;
        }
    }
}

__global__ void copyX(int nNodes, const Node* const* nodes, Vector3f* x) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads)
        x[i] = nodes[i]->x;
}

__global__ void copyV(int nNodes, const Node* const* nodes, Vector3f* v) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads)
        v[i] = nodes[i]->v;
}

__global__ void copyU(int nVertices, const Vertex* const* vertices, Vector2f* u) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nVertices; i += nThreads)
        u[i] = vertices[i]->u;
}

__global__ void copyFaceIndices(int nFaces, const Face* const* faces, Pairii* indices) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads) {
        const Face* face = faces[i];
        for (int j = 0; j < 3; j++) {
            const Vertex* vertex = face->vertices[j];
            Pairii& index = indices[3 * i + j];
            index.first = vertex->node->index;
            index.second = vertex->index;
        }
    }
}

__global__ void printDebugInfoGpu(const Face* const* faces, int index) {
    const Face* face = faces[index];
    printf("Nodes=[%d, %d, %d]\n", face->vertices[0]->node->index, face->vertices[1]->node->index, face->vertices[2]->node->index);
}

__global__ void checkEdges(int nEdges, const Edge* const* edges) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads) {
        const Edge* edge = edges[i];
        for (int j = 0; j < 2; j++)
            if (edge->opposites[j] != nullptr) {
                if (edge->vertices[j][0]->node != edge->nodes[0] || edge->vertices[j][1]->node != edge->nodes[1])
                    printf("Edge vertices check error!\n");
                if (edge->adjacents[j] == nullptr || !edge->adjacents[j]->contain(edge->opposites[j]) || !edge->adjacents[j]->contain(edge))
                    printf("Edge adjacents check error!\n");
            } else if (edge->adjacents[j] != nullptr)
                printf("Edge opposites check error!\n");
    }
}

__global__ void checkFaces(int nFaces, const Face* const* faces) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads) {
        const Face* face = faces[i];
        for (int j = 0; j < 3; j++) {
            Edge* edge = face->edges[j];
            if (edge->adjacents[0] != face && edge->adjacents[1] != face)
                printf("Face edges check error!\n");
        }
    }
}