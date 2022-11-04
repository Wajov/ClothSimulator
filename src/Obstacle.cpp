#include "Obstacle.hpp"

Obstacle::Obstacle(const Json::Value& json) {
    Transform* transform = new Transform(json["transform"]);
    mesh = new Mesh(json["mesh"], transform, nullptr);
    shader = new Shader("shader/VertexShader.glsl", "shader/FaceFragmentShader.glsl");
    delete transform;
}

Obstacle::~Obstacle() {
    delete mesh;
    delete shader;
}

Mesh* Obstacle::getMesh() const {
    return mesh;
}

void Obstacle::reset() const {
    mesh->reset();
}

void Obstacle::bind() {
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
    mesh->renderFace();
}
