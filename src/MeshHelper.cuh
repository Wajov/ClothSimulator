#ifndef MESH_HELPER_CUH
#define MESH_HELPER_CUH

#include <device_launch_parameters.h>
#include <thrust/pair.h>

#include "MathHelper.cuh"
#include "Node.cuh"
#include "Vertex.cuh"
#include "Edge.cuh"
#include "Face.cuh"
#include "Renderable.cuh"
#include "Material.cuh"

struct EdgeData {
    Vertex* opposite;
    Face* adjacent;
};

struct NodeData {
    float mass, area;

    __device__ NodeData& operator=(const NodeData& d) {
        mass = d.mass;
        area = d.area;
        return *this;
    };

    __device__ NodeData operator+(const NodeData& d) const {
        NodeData ans;
        ans.mass = mass + d.mass;
        ans.area = area + d.area;
        return ans;
    };
};

struct IsNull {
    template<typename T> __device__ bool operator()(const T* p) const {
        return p == nullptr;
    }
};

__global__ static void initializeNodes(int nNodes, const Vector3f* x, bool isFree, Node** nodes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads)
        nodes[i] = new Node(x[i], isFree);
}

__global__ static void initializeVertices(int nVertices, const Vector2f* u, Vertex** vertices) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nVertices; i += nThreads)
        vertices[i] = new Vertex(u[i]);
}

__device__ static void setEdgeData(int index0, int index1, const Vertex* vertex, const Face* face, PairIndex& index, EdgeData& edgeData) {
    if (index0 > index1)
        mySwap(index0, index1);
    
    index.first = index0;
    index.second = index1;
    edgeData.opposite = const_cast<Vertex*>(vertex);
    edgeData.adjacent = const_cast<Face*>(face);
}

__global__ static void initializeFaces(int nFaces, const int* xIndices, const int* uIndices, const Node* const* nodes, const Material* material, Vertex** vertices, Face** faces, PairIndex* edgeIndices, EdgeData* edgeData) {
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

__global__ static void initializeEdges(int nEdges, const PairIndex* indices, const EdgeData* edgeData, const Node* const* nodes, Edge** edges) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads)
        if (i == 0 || indices[i] != indices[i - 1]) {
            const PairIndex& index = indices[i];
            const EdgeData& e = edgeData[i];
            Edge* edge = new Edge(nodes[index.first], nodes[index.second]);
            edge->initialize(e.opposite, e.adjacent);
            e.adjacent->setEdge(edge);
            edges[i] = edge;
        } else
            edges[i] = nullptr;
}

__global__ static void setEdges(int nEdges, const PairIndex* indices, const EdgeData* edgeData, Edge** edges) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads)
        if (i > 0 && indices[i] == indices[i - 1]) {
            const EdgeData& e = edgeData[i];
            edges[i - 1]->initialize(e.opposite, e.adjacent);
        }
}

__global__ static void collectPreservedNodes(int nEdges, const Edge* const* edges, int* nodeIndices) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads) {
        const Edge* edge = edges[i];
        if (edge->isBoundary() || edge->isSeam())
            for (int j = 0; j < 2; j++)
                nodeIndices[2 * i + j] = edge->nodes[j]->index;
        else
            for (int j = 0; j < 2; j++)
                nodeIndices[2 * i + j] = -1;
    }
}

__global__ static void setPreservedNodes(int nIndices, const int* indices, Node** nodes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nIndices; i += nThreads) {
        int index = indices[i];
        if (index > -1)
            nodes[index]->preserve = true;
    }
}

__global__ static void resetGpu(int nNodes, Node** nodes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads) {
        Node* node = nodes[i];
        node->x = node->x0;
    }
}

__global__ static void updateNodeIndices(int nNodes, Node** nodes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads)
        nodes[i]->index = i;
}

__global__ static void updateVertexIndices(int nVertices, Vertex** vertices) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nVertices; i += nThreads)
        vertices[i]->index = i;
}

__global__ static void collectNodeStructures(int nFaces, Face** faces, int* indices, NodeData* nodeData) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads) {
        Face* face = faces[i];
        float mass = face->mass / 3.0f;
        float area = face->area;
        for (int j = 0; j < 3; j++) {
            int index = 3 * i + j;
            indices[index] = face->vertices[j]->node->index;
            nodeData[index].mass = mass;
            nodeData[index].area = area;
        }
    }
}

__global__ static void setNodeStructures(int nNodes, const int* indices, const NodeData* nodeData, Node** nodes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads) {
        Node* node = nodes[indices[i]];
        node->mass = nodeData[i].mass;
        node->area = nodeData[i].area;
    }
}

__global__ static void updateNodeGeometries(int nNodes, float invDt, Node** nodes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads) {
        Node* node = nodes[i];
        node->x1 = node->x;
        node->v = (node->x - node->x0) * invDt;
    }
}

__global__ static void updateEdgeGeometries(int nEdges, Edge** edges) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads)
        edges[i]->update();
}

__global__ static void updateFaceGeometries(int nFaces, Face** faces) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads)
        faces[i]->update();
}

__global__ static void collectNodeGeometries(int nFaces, Face** faces, int* indices, Vector3f* nodeData) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads) {
        Face* face = faces[i];
        for (int j = 0; j < 3; j++) {
            int index = 3 * i + j;
            Node* node = face->vertices[j]->node;
            Vector3f e0 = face->vertices[(j + 1) % 3]->node->x - node->x;
            Vector3f e1 = face->vertices[(j + 2) % 3]->node->x - node->x;
            indices[index] = node->index;
            nodeData[index] = e0.cross(e1) / (e0.norm2() * e1.norm2());
        }
    }
}

__global__ static void setNodeGeometries(int nNodes, const int* indices, const Vector3f* nodeData, Node** nodes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads)
        nodes[indices[i]]->n = nodeData[i].normalized();
}

__global__ static void updateRenderingDataGpu(int nFaces, const Face* const* faces, Renderable* renderables) {
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

__global__ static void printDebugInfoGpu(const Face* const* faces, int index) {
    const Face* face = faces[index];
    printf("Nodes=[%d, %d, %d]\n", face->vertices[0]->node->index, face->vertices[1]->node->index, face->vertices[2]->node->index);
}

__global__ static void deleteNodes(int nNodes, const Node* const* nodes) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nNodes; i += nThreads)
        delete nodes[i];
}

__global__ static void deleteVertices(int nVertices, const Vertex* const* vertices) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nVertices; i += nThreads)
        delete vertices[i];
}

__global__ static void deleteEdges(int nEdges, const Edge* const* edges) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nEdges; i += nThreads)
        delete edges[i];
}

__global__ static void deleteFaces(int nFaces, const Face* const* faces) {
    int nThreads = gridDim.x * blockDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nFaces; i += nThreads)
        delete faces[i];
}

#endif