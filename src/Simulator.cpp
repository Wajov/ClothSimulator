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

    cloths[0]->readDataFromFile("input.txt");
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
    for (Cloth* cloth : cloths)
        cloth->update();
}

void Simulator::findImpacts(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, std::vector<Impact>& impacts) const {
    for (int i = 0; i < clothBvhs.size(); i++) {
        clothBvhs[i]->findImpacts(COLLISION_THICKNESS, impacts);
        for (int j = 0; j < i; j++)
            clothBvhs[i]->findImpacts(clothBvhs[j], COLLISION_THICKNESS, impacts);
        
        for (int j = 0; j < obstacleBvhs.size(); j++)
            clothBvhs[i]->findImpacts(obstacleBvhs[j], COLLISION_THICKNESS, impacts);
    }
}

void Simulator::updateActive(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, const std::vector<ImpactZone*>& zones) const {
    for (BVH* clothBvh : clothBvhs)
        clothBvh->setAllActive(false);
    for (BVH* obstacleBvh : obstacleBvhs)
        obstacleBvh->setAllActive(false);
    
    for (ImpactZone* zone : zones) {
        if (!zone->getActive())
            continue;
        std::vector<Vertex*>& vertices = zone->getVertices();
        for (const Vertex* vertex : vertices) {
            for (BVH* clothBvh : clothBvhs)
                if (clothBvh->contain(vertex))
                    clothBvh->setActive(vertex);
            for (BVH* obstacleBvh : obstacleBvhs)
                if (obstacleBvh->contain(vertex))
                    obstacleBvh->setActive(vertex);
        }
    }
}

void Simulator::collisionStep() {
    std::vector<BVH*> clothBvhs, obstacleBvhs;
    for (const Cloth* cloth : cloths)
        clothBvhs.push_back(new BVH(cloth->getMesh(), true));
    for (const Obstacle* obstacle: obstacles)
        obstacleBvhs.push_back(new BVH(obstacle->getMesh(), true));

    // std::vector<Impact> impacts;
    // findImpacts(clothBvhs, obstacleBvhs, impacts);
    // std::ofstream fout("output_impacts.txt");
    // for (const Impact& impact : impacts) {
    //     for (int i = 0; i < 4; i++)
    //         fout << impact.vertices[i]->index << ' ';
    //     fout << std::endl;
    // }
    
    std::vector<ImpactZone*> zones;
    obstacleMass = 1e3f;
    for (int deform = 0; deform < 2; deform++) {
        zones.clear();
        bool success = false;
        for (int i = 0; i < MAXIMUM_ITERATION; i++) {
            if (!zones.empty())
                updateActive(clothBvhs, obstacleBvhs, zones);
            
            std::vector<Impact> impacts;
            findImpacts(clothBvhs, obstacleBvhs, impacts);
            impacts = std::move(independentImpacts(impacts));
            if (impacts.empty()) {
                success = true;
                break;
            }

            addImpacts(impacts, zones);
            for (const ImpactZone* zone : zones)
                if (zone->getActive()) {
                    Optimization optimization = new CollisionOptimization(zone);
                    augmentedLagrangianMethod(optimization);
                    delete optimization;
                }

            for (BVH* clothBvh : clothBvhs)
                clothBvh->update();
            for (BVH* obstacleBvh : obstacleBvhs)
                obstacleBvh->update();
            if (deform == 1)
                obstacleMass *= 0.5f;
        }
        if (success)
            break;
    }

    for (Cloth* cloth : cloths)
        cloth->update();

    for (const BVH* clothBvh : clothBvhs)
        delete clothBvh;
    for (const BVH* obstacleBvh : obstacleBvhs)
        delete obstacleBvh;
    for (const ImpactZone* zone : zones)
        delete zone;
}

void Simulator::render(const Matrix4x4f& model, const Matrix4x4f& view, const Matrix4x4f& projection, const Vector3f& cameraPosition, const Vector3f& lightPosition, float lightPower) const {
    for (const Cloth* cloth : cloths)
        cloth->render(model, view, projection, cameraPosition, lightPosition, lightPower);

    for (const Obstacle* obstacle : obstacles)
        obstacle->render(model, view, projection, cameraPosition, lightPosition, lightPower);
}

void Simulator::step() {
    // physicsStep();
    collisionStep();
    exit(0);
    // TODO

    for (Cloth* cloth : cloths)
        cloth->getMesh()->updateRenderingData();
    for (Obstacle* obstacle : obstacles)
        obstacle->getMesh()->updateRenderingData();
}
