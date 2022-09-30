#include "Simulator.hpp"

Simulator::Simulator(const std::string& path) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "Failed to open configuration file: " << path << std::endl;
        exit(1);
    }

    Json::Value json;
    fin >> json;

    frameTime = parseFloat(json["frame_time"]);
    frameSteps = parseFloat(json["frame_steps"]);
    dt =  frameTime / frameSteps;

    gravity = parseVector3f(json["gravity"]);
    wind = new Wind();
    
    for (const Json::Value& clothJson : json["cloths"])
        cloths.push_back(new Cloth(clothJson));

    for (const Json::Value& obstacleJson : json["obstacles"])
        obstacles.push_back(new Obstacle(obstacleJson));

    fin.close();

    // cloths[0]->readDataFromFile("input.txt");
}

Simulator::~Simulator() {
    delete wind;
    for (const Cloth* cloth : cloths)
        delete cloth;
    for (const Obstacle* obstacle : obstacles)
        delete obstacle;
}

void Simulator::physicsStep() {
    for (Cloth* cloth : cloths)
        cloth->physicsStep(dt, gravity, wind);
}

void Simulator::render(const Matrix4x4f& model, const Matrix4x4f& view, const Matrix4x4f& projection, const Vector3f& cameraPosition, const Vector3f& lightPosition, float lightPower) const {
    for (const Cloth* cloth : cloths)
        cloth->render(model, view, projection, cameraPosition, lightPosition, lightPower);

    for (const Obstacle* obstacle : obstacles)
        obstacle->render(model, view, projection, cameraPosition, lightPosition, lightPower);
}

void Simulator::step() {
    physicsStep();

}
