#include "Obstacle.cuh"

Obstacle::Obstacle(const Json::Value& json, const std::vector<Transformation>& transformations, MemoryPool* pool) {
    Transformation transformation(json["transform"]);
    motionIndex = json["motion"].isNull() ? -1 : parseInt(json["motion"]);
    mesh = new Mesh(parseString(json["mesh"]), transformation, nullptr, pool);

    initialize();
    transform(transformations);
}

Obstacle::Obstacle(const std::string& path, int motionIndex, const std::vector<Transformation>& transformations, MemoryPool* pool) :
    motionIndex(motionIndex),
    mesh(new Mesh(path, Transformation(), nullptr, pool)) {
    initialize();
    transform(transformations);
}

Obstacle::~Obstacle() {
    delete mesh;
    delete shader;
}

void Obstacle::initialize() {
    if (!gpu) {
        std::vector<Node*>& nodes = mesh->getNodes();
        int nNodes = nodes.size();
        base.resize(nNodes);
        for (int i = 0; i < nNodes; i++)
            base[i] = nodes[i]->x;
    } else {
        thrust::device_vector<Node*>& nodes = mesh->getNodesGpu();
        int nNodes = nodes.size();
        baseGpu.resize(nNodes);
        setBase<<<GRID_SIZE, BLOCK_SIZE>>>(nNodes, pointer(nodes), pointer(baseGpu));
        CUDA_CHECK_LAST();
    }
}

Mesh* Obstacle::getMesh() const {
    return mesh;
}

void Obstacle::transform(const std::vector<Transformation>& transformations) {
    if (motionIndex > -1) {
        Transformation transformation = transformations[motionIndex];
        if (!gpu) {
            std::vector<Node*>& nodes = mesh->getNodes();

            for (int i = 0; i < nodes.size(); i++) {
                Node* node = nodes[i];
                node->x0 = node->x = transformation.applyToPoint(base[i]);
            }
        } else {
            thrust::device_vector<Node*>& nodes = mesh->getNodesGpu();

            transformGpu<<<GRID_SIZE, BLOCK_SIZE>>>(nodes.size(), pointer(baseGpu), transformation, pointer(nodes));
            CUDA_CHECK_LAST();
        }

        mesh->updateFaceGeometries();
        mesh->updateNodeGeometries();
    }
}

void Obstacle::step(float dt, const std::vector<Transformation>& transformations) {
    if (motionIndex > -1) {
        float invDt = 1.0f / dt;
        Transformation transformation = transformations[motionIndex];
        if (!gpu) {
            std::vector<Node*>& nodes = mesh->getNodes();

            for (int i = 0; i < nodes.size(); i++) {
                Node* node = nodes[i];
                Vector3f x = transformation.applyToPoint(base[i]);
                node->v = (x - node->x) * invDt;
            }
        } else {
            thrust::device_vector<Node*>& nodes = mesh->getNodesGpu();

            stepGpu<<<GRID_SIZE, BLOCK_SIZE>>>(nodes.size(), invDt, pointer(baseGpu), transformation, pointer(nodes));
            CUDA_CHECK_LAST();
        }
    }
}

void Obstacle::bind() {
    shader = new Shader("shader/Vertex.glsl", "shader/FaceFragment.glsl");
    mesh->bind();
}

void Obstacle::render(const Matrix4x4f& model, const Matrix4x4f& view, const Matrix4x4f& projection, const Vector3f& cameraPosition, const Vector3f& lightDirection) const {
    shader->use();
    shader->setMat4("model", model);
    shader->setMat4("view", view);
    shader->setMat4("projection", projection);
    shader->setVec3("color", Vector3f(0.8f, 0.8f, 0.8f));
    shader->setVec3("cameraPosition", cameraPosition);
    shader->setVec3("lightDirection", lightDirection);
    mesh->render();
}