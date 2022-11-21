#include "Simulator.cuh"

Simulator::Simulator(const std::string& path) :
    MAX_ITERATION(30),
    nSteps(0) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "Failed to open configuration file: " << path << std::endl;
        exit(1);
    }

    Json::Value json;
    fin >> json;

    frameTime = parseFloat(json["frame_time"]);
    frameSteps = parseInt(json["frame_steps"]);
    dt =  frameTime / frameSteps;

    gravity = parseVector3f(json["gravity"]);
    wind = new Wind();

    magic = new Magic(json["magic"]);
    
    for (const Json::Value& clothJson : json["cloths"])
        cloths.push_back(new Cloth(clothJson));

    for (const Json::Value& obstacleJson : json["obstacles"])
        obstacles.push_back(new Obstacle(obstacleJson));

    fin.close();

    // cloths[0]->readDataFromFile("input.txt");
    remeshingStep();
    bind();

    if (gpu) {
        CUDA_CHECK(cudaMalloc(&windGpu, sizeof(Wind)));
        CUDA_CHECK(cudaMemcpy(windGpu, wind, sizeof(Wind), cudaMemcpyHostToDevice));
    }
}

Simulator::~Simulator() {
    delete magic;
    delete wind;
    for (const Cloth* cloth : cloths)
        delete cloth;
    for (const Obstacle* obstacle : obstacles)
        delete obstacle;

    if (gpu)
        CUDA_CHECK(cudaFree(windGpu));
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

void Simulator::traverse(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, float thickness, std::function<void(const Face*, const Face*, float)> callback) {
    for (int i = 0; i < clothBvhs.size(); i++) {
        clothBvhs[i]->traverse(thickness, callback);
        for (int j = 0; j < i; j++)
            clothBvhs[i]->traverse(clothBvhs[j], thickness, callback);
        
        for (int j = 0; j < obstacleBvhs.size(); j++)
            clothBvhs[i]->traverse(obstacleBvhs[j], thickness, callback);
    }
}

std::vector<Impact> Simulator::independentImpacts(const std::vector<Impact>& impacts) const {
    std::vector<Impact> sorted = impacts;
    std::sort(sorted.begin(), sorted.end());
    
    std::unordered_set<Vertex*> vertices;
    std::vector<Impact> ans;
    for (const Impact& impact : sorted) {
        bool flag = true;
        for (int i = 0; i < 4; i++)
            if (impact.vertices[i]->isFree && vertices.find(impact.vertices[i]) != vertices.end()) {
                flag = false;
                break;
            }
        if (flag) {
            ans.push_back(impact);
            for (int i = 0; i < 4; i++)
                vertices.insert(impact.vertices[i]);
        }
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
                    if (zone != zoneTemp) {
                        zone->merge(zoneTemp);
                        zones.erase(std::remove(zones.begin(), zones.end(), zoneTemp), zones.end());
                        delete zoneTemp;
                    }
                }
            }
        }
        zone->addImpact(impact);
        zone->setActive(true);
    }
}

void Simulator::resetObstacles() {
    for (Obstacle* obstacle : obstacles)
        obstacle->reset();
}

void Simulator::physicsStep() {
    for (Cloth* cloth : cloths)
        cloth->physicsStep(dt, magic->handleStiffness, gravity, !gpu ? wind : windGpu);
    updateGeometries();
}

void Simulator::collisionStep() {
    if (!gpu) {
        std::vector<BVH*> clothBvhs, obstacleBvhs;
        for (const Cloth* cloth : cloths)
            clothBvhs.push_back(new BVH(cloth->getMesh(), true));
        for (const Obstacle* obstacle: obstacles)
            obstacleBvhs.push_back(new BVH(obstacle->getMesh(), true));
        
        std::vector<ImpactZone*> zones;
        float obstacleMass = 1e3f;
        for (int deform = 0; deform < 2; deform++) {
            zones.clear();
            bool success = false;
            for (int i = 0; i < MAX_ITERATION; i++) {
                if (!zones.empty())
                    updateActive(clothBvhs, obstacleBvhs, zones);
                
                std::vector<Impact> impacts;
                traverse(clothBvhs, obstacleBvhs, magic->collisionThickness, [&](const Face* face0, const Face* face1, float thickness) {
                    checkImpacts(face0, face1, thickness, impacts);
                });
                impacts = std::move(independentImpacts(impacts));
                if (impacts.empty()) {
                    success = true;
                    break;
                }

                addImpacts(impacts, zones, deform == 1);
                for (const ImpactZone* zone : zones)
                    if (zone->getActive()) {
                        Optimization* optimization = new ImpactZoneOptimization(zone, magic->collisionThickness, obstacleMass);
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

        updateGeometries();
        updateVelocities();

        for (const BVH* clothBvh : clothBvhs)
            delete clothBvh;
        for (const BVH* obstacleBvh : obstacleBvhs)
            delete obstacleBvh;
        for (const ImpactZone* zone : zones)
            delete zone;
    } else {
        // TODO
    }
}

void Simulator::remeshingStep() {
    if (!gpu) {
        std::vector<BVH*> obstacleBvhs;
        for (const Obstacle* obstacle: obstacles)
            obstacleBvhs.push_back(new BVH(obstacle->getMesh(), false));

        for (Cloth* cloth : cloths)
            cloth->remeshingStep(obstacleBvhs, 10.0f * magic->repulsionThickness);

        updateIndices();
        updateGeometries();

        for (const BVH* obstacleBvh : obstacleBvhs)
            delete obstacleBvh;
    } else {
        // TODO
    }
}

void Simulator::updateGeometries() {
    for (Cloth* cloth : cloths)
        cloth->updateGeometries();
}

void Simulator::updateVelocities() {
    for (Cloth* cloth : cloths)
        cloth->updateVelocities(dt);
}

void Simulator::updateIndices() {
    for (Cloth* cloth : cloths)
        cloth->updateIndices();
}

void Simulator::updateRenderingData(bool rebind) {
    for (Cloth* cloth : cloths)
        cloth->updateRenderingData(rebind);
}

void Simulator::bind() {
    for (Cloth* cloth : cloths)
        cloth->bind();
    for (Obstacle* obstacle : obstacles)
        obstacle->bind();
}

void Simulator::render(const Matrix4x4f& model, const Matrix4x4f& view, const Matrix4x4f& projection, const Vector3f& cameraPosition, const Vector3f& lightDirection) const {
    for (const Cloth* cloth : cloths)
        cloth->render(model, view, projection, cameraPosition, lightDirection);

    for (const Obstacle* obstacle : obstacles)
        obstacle->render(model, view, projection, cameraPosition, lightDirection);
}

void Simulator::step() {
    nSteps++;
    std::cout << "Step [" << nSteps << "]:" << std::endl;

    resetObstacles();
    
    std::chrono::duration<float> d;
    auto t0 = std::chrono::high_resolution_clock::now();
    
    physicsStep();
    auto t1 = std::chrono::high_resolution_clock::now();
    d = t1 - t0;
    std::cout << "Physics Step: " << d.count() << "s";
    
    collisionStep();
    auto t2 = std::chrono::high_resolution_clock::now();
    d = t2 - t1;
    std::cout << ", Collision Step: " << d.count() << "s";
    
    if (nSteps % frameSteps == 0) {
        remeshingStep();
        auto t3 = std::chrono::high_resolution_clock::now();
        d = t3 - t2;
        std::cout << ", Remeshing Step: " << d.count() << "s";
        updateRenderingData(true);
    } else
        updateRenderingData(false);

    std::cout << std::endl;
}
