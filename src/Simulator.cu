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

void Simulator::traverse(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, float thickness, std::function<void(const Face*, const Face*, float)> callback) const {
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

void Simulator::updateActive(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, const std::vector<Impact>& impacts) const {
    for (BVH* clothBvh : clothBvhs)
        clothBvh->setAllActive(false);
    for (BVH* obstacleBvh : obstacleBvhs)
        obstacleBvh->setAllActive(false);
    
    for (const Impact& impact : impacts)
        for (int i = 0; i < 4; i++) {
            Node* node = impact.nodes[i];
            for (BVH* clothBvh : clothBvhs)
                if (clothBvh->contain(node))
                    clothBvh->setActive(node, true);
            for (BVH* obstacleBvh : obstacleBvhs)
                if (obstacleBvh->contain(node))
                    obstacleBvh->setActive(node, true);
        }
}

void Simulator::updateActive(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, const std::vector<Intersection>& intersections) const {
    for (BVH* clothBvh : clothBvhs)
        clothBvh->setAllActive(false);
    for (BVH* obstacleBvh : obstacleBvhs)
        obstacleBvh->setAllActive(false);
    
    for (const Intersection& intersection : intersections)
        for (int i = 0; i < 3; i++) {
            Node* node0 = intersection.face0->vertices[i]->node;
            for (BVH* clothBvh : clothBvhs)
                if (clothBvh->contain(node0))
                    clothBvh->setActive(node0, true);
            for (BVH* obstacleBvh : obstacleBvhs)
                if (obstacleBvh->contain(node0))
                    obstacleBvh->setActive(node0, true);
            
            Node* node1 = intersection.face1->vertices[i]->node;
            for (BVH* clothBvh : clothBvhs)
                if (clothBvh->contain(node1))
                    clothBvh->setActive(node1, true);
            for (BVH* obstacleBvh : obstacleBvhs)
                if (obstacleBvh->contain(node1))
                    obstacleBvh->setActive(node1, true);
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
    float obstacleMass = 1e3f;

    if (!gpu) {
        std::vector<Impact> impacts;
        for (int deform = 0; deform < 2; deform++) {
            impacts.clear();
            bool success = false;
            for (int i = 0; i < MAX_COLLISION_ITERATION; i++) {
                if (!impacts.empty())
                    updateActive(clothBvhs, obstacleBvhs, impacts);
                
                std::vector<Impact> newImpacts;
                traverse(clothBvhs, obstacleBvhs, magic->collisionThickness, [&](const Face* face0, const Face* face1, float thickness) {
                    checkImpacts(face0, face1, thickness, newImpacts);
                });
                newImpacts = std::move(independentImpacts(newImpacts));
                if (newImpacts.empty()) {
                    success = true;
                    break;
                }

                impacts.insert(impacts.end(), newImpacts.begin(), newImpacts.end());
                Optimization* optimization = new CollisionOptimization(impacts, magic->collisionThickness, deform, obstacleMass);
                augmentedLagrangianMethod(optimization);
                delete optimization;

                updateBvhs(clothBvhs);
                if (deform == 1) {
                    updateBvhs(obstacleBvhs);
                    obstacleMass *= 0.5f;
                }
            }
            if (success)
                break;
        }
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
                updateActive(clothBvhs, obstacleBvhs, intersections);
            
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
