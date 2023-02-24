#include "Simulator.cuh"

Simulator::Simulator(const std::string& path) :
    nSteps(0),
    selectedCloth(-1) {
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
    Wind* windTemp = new Wind();

    magic = new Magic(json["magic"]);
    
    cloths.resize(json["cloths"].size());
    for (int i = 0; i < json["cloths"].size(); i++)
        cloths[i] = new Cloth(json["cloths"][i]);

    obstacles.resize(json["obstacles"].size());
    for (int i = 0; i < json["obstacles"].size(); i++)
        obstacles[i] = new Obstacle(json["obstacles"][i]);

    fin.close();

    // cloths[0]->readDataFromFile("input.txt");
    remeshingStep();
    bind();

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenTextures(1, &indexTexture);
    glGenRenderbuffers(1, &rbo);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    indexShader = new Shader("shader/Vertex.glsl", "shader/IndexFragment.glsl");

    if (!gpu)
        wind = windTemp;
    else {
        CUDA_CHECK(cudaMalloc(&wind, sizeof(Wind)));
        CUDA_CHECK(cudaMemcpy(wind, windTemp, sizeof(Wind), cudaMemcpyHostToDevice));
        delete windTemp;
    }
}

Simulator::~Simulator() {
    delete magic;
    for (const Cloth* cloth : cloths)
        delete cloth;
    for (const Obstacle* obstacle : obstacles)
        delete obstacle;
    delete indexShader;

    if (!gpu)
        delete wind;
    else
        CUDA_CHECK(cudaFree(wind));
}

std::vector<BVH*> Simulator::buildClothBvhs(bool ccd) const {
    std::vector<BVH*> ans(cloths.size());
    for (int i = 0; i < cloths.size(); i++)
        ans[i] = new BVH(cloths[i]->getMesh(), ccd);
    return ans;
}

std::vector<BVH*> Simulator::buildObstacleBvhs(bool ccd) const {
    std::vector<BVH*> ans(obstacles.size());
    for (int i = 0; i < obstacles.size(); i++)
        ans[i] = new BVH(obstacles[i]->getMesh(), ccd);
    return ans;
}

void Simulator::updateBvhs(std::vector<BVH*>& bvhs) const {
    for (BVH* bvh : bvhs)
        bvh->update();
}

void Simulator::destroyBvhs(const std::vector<BVH*>& bvhs) const {
    for (const BVH* bvh : bvhs)
        delete bvh;
}

void Simulator::traverse(const BVHNode* node, float thickness, std::function<void(const Face*, const Face*, float)> callback) {
    if (node->isLeaf() || !node->active)
        return;

    traverse(node->left, thickness, callback);
    traverse(node->right, thickness, callback);
    traverse(node->left, node->right, thickness, callback);
}

void Simulator::traverse(const BVH* bvh, float thickness, std::function<void(const Face*, const Face*, float)> callback) {
    traverse(bvh->getRoot(), thickness, callback);
}

void Simulator::traverse(const BVH* bvh0, const BVH* bvh1, float thickness, std::function<void(const Face*, const Face*, float)> callback) {
    traverse(bvh0->getRoot(), bvh1->getRoot(), thickness, callback);
}

void Simulator::traverse(const BVHNode* node0, const BVHNode* node1, float thickness, std::function<void(const Face*, const Face*, float)> callback) {
    if (!node0->active && !node1->active)
        return;
    if (!node0->bounds.overlap(node1->bounds, thickness))
        return;

    if (node0->isLeaf() && node1->isLeaf())
        callback(node0->face, node1->face, thickness);
    else if (node0->isLeaf()) {
        traverse(node0, node1->left, thickness, callback);
        traverse(node0, node1->right, thickness, callback);
    } else {
        traverse(node0->left, node1, thickness, callback);
        traverse(node0->right, node1, thickness, callback);
    }
}

void Simulator::traverse(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, float thickness, std::function<void(const Face*, const Face*, float)> callback) {
    for (int i = 0; i < clothBvhs.size(); i++) {
        traverse(clothBvhs[i], thickness, callback);
        for (int j = 0; j < i; j++)
            traverse(clothBvhs[i], clothBvhs[j], thickness, callback);
        
        for (int j = 0; j < obstacleBvhs.size(); j++)
            traverse(clothBvhs[i], obstacleBvhs[j], thickness, callback);
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
        std::vector<Node*>& nodes = zone->getNodes();
        for (const Node* node : nodes) {
            for (BVH* clothBvh : clothBvhs)
                if (clothBvh->contain(node))
                    clothBvh->setActive(node, true);
            for (BVH* obstacleBvh : obstacleBvhs)
                if (obstacleBvh->contain(node))
                    obstacleBvh->setActive(node, true);
        }
    }
}

std::vector<Impact> Simulator::independentImpacts(const std::vector<Impact>& impacts) const {
    std::vector<Impact> sorted = impacts;
    std::sort(sorted.begin(), sorted.end());
    
    std::unordered_set<Node*> nodes;
    std::vector<Impact> ans;
    for (const Impact& impact : sorted) {
        bool flag = true;
        for (int i = 0; i < 4; i++)
            if (impact.nodes[i]->isFree && nodes.find(impact.nodes[i]) != nodes.end()) {
                flag = false;
                break;
            }
        if (flag) {
            ans.push_back(impact);
            for (int i = 0; i < 4; i++)
                nodes.insert(impact.nodes[i]);
        }
    }
    return ans;
}

ImpactZone* Simulator::findImpactZone(const Node* node, std::vector<ImpactZone*>& zones) const {
    for (ImpactZone* zone : zones)
        if (zone->contain(node))
            return zone;

    ImpactZone* zone = new ImpactZone();
    zone->addNode(node);
    zones.push_back(zone);
    return zone;
}

void Simulator::addImpacts(const std::vector<Impact>& impacts, std::vector<ImpactZone*>& zones, bool deformObstacles) const {
    for (ImpactZone* zone : zones)
        zone->setActive(false);
    
    for (const Impact& impact : impacts) {
        ImpactZone* zone = nullptr;
        for (int i = 0; i < 4; i++) {
            Node* node = impact.nodes[i];
            if (node->isFree || deformObstacles) {
                if (zone == nullptr)
                    zone = findImpactZone(node, zones);
                else {
                    ImpactZone* zoneTemp = findImpactZone(node, zones);
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

void Simulator::updateActive(const std::vector<BVH*>& clothBvhs, const std::vector<Intersection>& intersections) const {
    for (BVH* clothBvh : clothBvhs)
        clothBvh->setAllActive(false);
    
    for (const Intersection& intersection : intersections)
        for (int i = 0; i < 3; i++) {
            Node* node0 = intersection.face0->vertices[i]->node;
            for (BVH* clothBvh : clothBvhs)
                if (clothBvh->contain(node0))
                    clothBvh->setActive(node0, true);
            
            Node* node1 = intersection.face1->vertices[i]->node;
            for (BVH* clothBvh : clothBvhs)
                if (clothBvh->contain(node1))
                    clothBvh->setActive(node1, true);
        }
}

void Simulator::resetObstacles() {
    for (Obstacle* obstacle : obstacles)
        obstacle->reset();
}

void Simulator::physicsStep() {
    for (Cloth* cloth : cloths)
        cloth->physicsStep(dt, magic->handleStiffness, gravity, wind);
    updateGeometries();
}

void Simulator::collisionStep() {
    std::vector<BVH*> clothBvhs = std::move(buildClothBvhs(true));
    std::vector<BVH*> obstacleBvhs = std::move(buildObstacleBvhs(true));

    if (!gpu) {
        std::vector<ImpactZone*> zones;
        float obstacleMass = 1e3f;
        for (int deform = 0; deform < 2; deform++) {
            zones.clear();
            bool success = false;
            for (int i = 0; i < MAX_COLLISION_ITERATION; i++) {
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

                updateBvhs(clothBvhs);
                updateBvhs(obstacleBvhs);
                if (deform == 1)
                    obstacleMass *= 0.5f;
            }
            if (success)
                break;
        }

        for (const ImpactZone* zone : zones)
            delete zone;
    } else {
        // TODO
    }

    destroyBvhs(clothBvhs);
    destroyBvhs(obstacleBvhs);

    updateGeometries();
    updateVelocities();
}

void Simulator::remeshingStep() {
    if (!gpu) {
        std::vector<BVH*> obstacleBvhs = std::move(buildObstacleBvhs(false));
        for (Cloth* cloth : cloths)
            cloth->remeshingStep(obstacleBvhs, 10.0f * magic->repulsionThickness);

        destroyBvhs(obstacleBvhs);
    } else {
        // TODO
    }

    updateStructures();
    updateGeometries();
}

void Simulator::separationStep(const std::vector<Mesh*>& oldMeshes) {
    if (!gpu) {
        std::vector<BVH*> clothBvhs = std::move(buildClothBvhs(false));
        std::vector<BVH*> obstacleBvhs = std::move(buildObstacleBvhs(false));
        std::vector<Intersection> intersections;
        
        for (int i = 0; i < MAX_SEPARATION_ITERATION; i++) {
            if (!intersections.empty())
                updateActive(clothBvhs, intersections);
            
            std::vector<Intersection> newIntersections;
            traverse(clothBvhs, obstacleBvhs, magic->collisionThickness, [&](const Face* face0, const Face* face1, float thickness) {
                checkIntersection(face0, face1, newIntersections, cloths, oldMeshes);
            });
            if (newIntersections.empty())
                break;

            intersections.insert(intersections.end(), newIntersections.begin(), newIntersections.end());
            Optimization* optimization = new SeparationOptimization(intersections, magic->collisionThickness);
            augmentedLagrangianMethod(optimization);
            delete optimization;

            updateBvhs(clothBvhs);
        }

        destroyBvhs(clothBvhs);
        destroyBvhs(obstacleBvhs);
    } else {
        // TODO
    }

    updateGeometries();
    updateVelocities();
}

void Simulator::updateStructures() {
    for (Cloth* cloth : cloths)
        cloth->updateStructures();
}

void Simulator::updateGeometries() {
    for (Cloth* cloth : cloths)
        cloth->updateGeometries();
}

void Simulator::updateVelocities() {
    for (Cloth* cloth : cloths)
        cloth->updateVelocities(dt);
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

void Simulator::render(int width, int height, const Matrix4x4f& model, const Matrix4x4f& view, const Matrix4x4f& projection, const Vector3f& cameraPosition, const Vector3f& lightDirection) const {
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
    glBindTexture(GL_TEXTURE_2D, indexTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32I, width, height, 0, GL_RG_INTEGER, GL_INT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, indexTexture, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, width, height);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);
    int color[2] = {-1, -1};
    glClearBufferiv(GL_COLOR, 0, color);
    glClear(GL_DEPTH_BUFFER_BIT);
    indexShader->use();
    indexShader->setMat4("model", model);
    indexShader->setMat4("view", view);
    indexShader->setMat4("projection", projection);
    for (int i = 0; i < cloths.size(); i++) {
        indexShader->setInt("clothIndex", i);
        cloths[i]->getMesh()->render();
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    for (int i = 0; i < cloths.size(); i++)
        cloths[i]->render(model, view, projection, cameraPosition, lightDirection, selectedCloth == i ? selectedFace : -1);
    
    for (const Obstacle* obstacle : obstacles)
        obstacle->render(model, view, projection, cameraPosition, lightDirection);
}

void Simulator::step() {
    nSteps++;
    std::cout << "Step [" << nSteps << "]:" << std::endl;

    selectedCloth = -1;

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
        std::vector<Mesh*> oldMeshes(cloths.size());
        for (int i = 0; i < cloths.size(); i++)
            oldMeshes[i] = new Mesh(cloths[i]->getMesh());

        remeshingStep();
        auto t3 = std::chrono::high_resolution_clock::now();
        d = t3 - t2;
        std::cout << ", Remeshing Step: " << d.count() << "s";

        separationStep(oldMeshes);
        auto t4 = std::chrono::high_resolution_clock::now();
        d = t4 - t3;
        std::cout << ", Separation Step: " << d.count() << "s";

        for (const Mesh* mesh : oldMeshes)
            delete mesh;

        updateRenderingData(true);
    } else
        updateRenderingData(false);

    std::cout << std::endl;
}

void Simulator::printDebugInfo(int x, int y) {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);

    Pixel pixel;
    glReadPixels(x, y, 1, 1, GL_RG_INTEGER, GL_INT, &pixel);
    selectedCloth = pixel.clothIndex;
    selectedFace = pixel.faceInedx;
    if (selectedCloth != -1 && selectedFace != -1)
        cloths[selectedCloth]->printDebugInfo(selectedFace);

    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
