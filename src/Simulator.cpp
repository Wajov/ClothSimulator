#include "Simulator.hpp"

Simulator::Simulator(const std::string& path) :
    COLLISION_THICKNESS(1e-4f),
    MAX_ITERATION(30) {
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
                    clothBvh->setActive(vertex, true);
            for (BVH* obstacleBvh : obstacleBvhs)
                if (obstacleBvh->contain(vertex))
                    obstacleBvh->setActive(vertex, true);
        }
    }
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

std::vector<Impact> Simulator::independentImpacts(const std::vector<Impact>& impacts) const {
    std::vector<Impact> sorted = impacts;
    std::sort(sorted.begin(), sorted.end());
    
    std::vector<Impact> ans;
    for (const Impact& impact : sorted) {
        bool conflict = false;
        for (const Impact& independentImpact : ans)
            if (impact.conflict(independentImpact)) {
                conflict = true;
                break;
            }
        if (!conflict)
            ans.push_back(impact);
    }
    return ans;
}

ImpactZone* Simulator::findImpactZone(const Vertex* vertex, std::vector<ImpactZone*>& zones) const {
    for (ImpactZone* zone : zones)
        if (zone->contain(vertex))
            return zone;

    ImpactZone* zone = new ImpactZone();
    zone->addVertex(vertex);
    zones.push_back(zone);
    return zone;
}

void Simulator::addImpacts(const std::vector<Impact>& impacts, std::vector<ImpactZone*>& zones, bool deformObstacles) const {
    for (ImpactZone* zone : zones)
        zone->setActive(false);
    
    for (const Impact& impact : impacts) {
        ImpactZone* zone = nullptr;
        for (int i = 0; i < 4; i++) {
            Vertex* vertex = impact.vertices[i];
            if (vertex->isFree || deformObstacles) {
                if (zone == nullptr)
                    zone = findImpactZone(vertex, zones);
                else {
                    ImpactZone* zoneTemp = findImpactZone(vertex, zones);
                    if (zone != zoneTemp)
                        zone->merge(zoneTemp);
                    std::remove(zones.begin(), zones.end(), zoneTemp);
                    delete zoneTemp;
                }
            }
        }
        zone->addImpact(impact);
        zone->setActive(true);
    }
}

void Simulator::physicsStep() {
    for (Cloth* cloth : cloths)
        cloth->physicsStep(dt, gravity, wind);
    for (Cloth* cloth : cloths)
        cloth->update();
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
    float obstacleMass = 1e3f;
    for (int deform = 0; deform < 2; deform++) {
        zones.clear();
        bool success = false;
        for (int i = 0; i < MAX_ITERATION; i++) {
            if (!zones.empty())
                updateActive(clothBvhs, obstacleBvhs, zones);
            
            std::vector<Impact> impacts;
            findImpacts(clothBvhs, obstacleBvhs, impacts);
            impacts = std::move(independentImpacts(impacts));
            if (impacts.empty()) {
                success = true;
                break;
            }

            addImpacts(impacts, zones, deform == 1);
            for (const ImpactZone* zone : zones)
                if (zone->getActive()) {
                    Optimization* optimization = new ImpactZoneOptimization(zone);
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
