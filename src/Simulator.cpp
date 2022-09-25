#include "Simulator.hpp"

Simulator::Simulator(const std::string& path) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "Failed to open configuration file: " << path << std::endl;
        exit(1);
    }

    Json::Value json;
    fin >> json;

    dt = json["frame_time"].asDouble() / json["frame_steps"].asDouble();
    for (int i = 0; i < 3; i++)
        gravity(i) = json["gravity"][i].asDouble();
    for (const Json::Value& clothJson : json["cloths"])
        cloths.push_back(new Cloth(clothJson));
    for (const Json::Value& handleJson : json["handles"])
        for (const Json::Value& nodeJson : handleJson["nodes"]) {
            int index = nodeJson.asInt();
            for (Cloth* cloth : cloths) {
                int size = cloth->getMesh()->getVertices().size();
                if (index < size) {
                    cloth->addHandle(index);
                    break;
                } else
                    index -= size;
            }
        }
    fin.close();

    wind = new Wind();

    cloths[0]->readDataFromFile("input.txt");
}

Simulator::~Simulator() {
    for (const Cloth* cloth : cloths)
        delete cloth;
}

void Simulator::renderEdge() const {
    for (Cloth* cloth : cloths)
        cloth->renderEdge();
}

void Simulator::renderFace() const {
    for (Cloth* cloth : cloths)
        cloth->renderFace();
}

void Simulator::update() {
    for (Cloth* cloth : cloths)
        cloth->update(dt, gravity, wind);
}
