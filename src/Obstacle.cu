#include "Obstacle.cuh"

Obstacle::Obstacle(const Json::Value& json, const std::vector<Motion*>& motions) {
    Transformation transformation(json["transform"]);
    mesh = new Mesh(json["mesh"], transformation, nullptr);

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

    motion = json["motion"].isNull() ? nullptr : motions[parseInt(json["motion"])];
    transform(0.0f);
}

Obstacle::~Obstacle() {
    delete mesh;
    delete shader;
}

Mesh* Obstacle::getMesh() const {
    return mesh;
}

void Obstacle::transform(float time) {
    if (motion != nullptr) {
        Transformation transformation = motion->computeTransformation(time);
        if (!gpu) {
            std::vector<Node*>& nodes = mesh->getNodes();

            for (int i = 0; i < nodes.size(); i++) {
                Node* node = nodes[i];
                node->x0 = node->x;
                node->x = transformation.applyToPoint(base[i]);
            }
        } else {
            thrust::device_vector<Node*>& nodes = mesh->getNodesGpu();
            
            transformGpu<<<GRID_SIZE, BLOCK_SIZE>>>(nodes.size(), pointer(baseGpu), transformation, pointer(nodes));
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