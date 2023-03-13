#include "Simulator.cuh"

Simulator::Simulator(SimulationMode mode, const std::string& path, const std::string& directory) :
    mode(mode),
    nSteps(0),
    nFrames(0),
    directory(directory) {
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1 << 30);
    if (mode == Simulate || mode == Resume || mode == Replay)
        renderer = new Renderer(900, 900);

    std::ifstream fin(mode == Simulate || mode == SimulateOffline ? path : directory + "/config.json");
    if (!fin.is_open()) {
        std::cerr << "Failed to open configuration file: " << path << std::endl;
        exit(1);
    }

    fin >> json;

    frameTime = parseFloat(json["frame_time"]);
    frameSteps = parseInt(json["frame_steps"]);
    dt =  frameTime / frameSteps;

    gravity = parseVector3f(json["gravity"]);
    Wind* windTemp = new Wind();
    if (!gpu)
        wind = windTemp;
    else {
        CUDA_CHECK(cudaMalloc(&wind, sizeof(Wind)));
        CUDA_CHECK(cudaMemcpy(wind, windTemp, sizeof(Wind), cudaMemcpyHostToDevice));
        delete windTemp;
    }
    
    magic = new Magic(json["magic"]);
    
    cloths.resize(json["cloths"].size());
    for (int i = 0; i < json["cloths"].size(); i++)
        cloths[i] = new Cloth(json["cloths"][i]);

    obstacles.resize(json["obstacles"].size());
    for (int i = 0; i < json["obstacles"].size(); i++)
        obstacles[i] = new Obstacle(json["obstacles"][i]);

    fin.close();
}

Simulator::~Simulator() {
    delete magic;
    for (const Cloth* cloth : cloths)
        delete cloth;
    for (const Obstacle* obstacle : obstacles)
        delete obstacle;
    delete renderer;

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

thrust::device_vector<Proximity> Simulator::traverse(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, float thickness) const {
    thrust::device_vector<Proximity> ans, proximities;
    for (int i = 0; i < cloths.size(); i++) {
        proximities = std::move(clothBvhs[i]->traverse(thickness));
        ans.insert(ans.end(), proximities.begin(), proximities.end());

        for (int j = 0; j < i; j++) {
            proximities = std::move(clothBvhs[i]->traverse(clothBvhs[j], thickness));
            ans.insert(ans.end(), proximities.begin(), proximities.end());
        }

        for (int j = 0; j < obstacleBvhs.size(); j++) {
            proximities = std::move(clothBvhs[i]->traverse(obstacleBvhs[j], thickness));
            ans.insert(ans.end(), proximities.begin(), proximities.end());
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

void Simulator::checkImpacts(const Face* face0, const Face* face1, float thickness, std::vector<Impact>& impacts) const {
    Impact impact;
    for (int i = 0; i < 3; i++)
        if (checkVertexFaceImpact(face0->vertices[i], face1, thickness, impact))
            impacts.push_back(impact);
    for (int i = 0; i < 3; i++)
        if (checkVertexFaceImpact(face1->vertices[i], face0, thickness, impact))
            impacts.push_back(impact);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (checkEdgeEdgeImpact(face0->edges[i], face1->edges[j], thickness, impact))
                impacts.push_back(impact);
}

thrust::device_vector<Impact> Simulator::findImpacts(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs) const {
    thrust::device_vector<Proximity> proximities = std::move(traverse(clothBvhs, obstacleBvhs, magic->collisionThickness));
    int nProximities = proximities.size();
    thrust::device_vector<Impact> ans(15 * nProximities);
    checkImpactsGpu<<<GRID_SIZE, BLOCK_SIZE>>>(nProximities, pointer(proximities), magic->collisionThickness, pointer(ans));
    CUDA_CHECK_LAST();

    ans.erase(thrust::remove_if(ans.begin(), ans.end(), IsNull()), ans.end());
    return ans;
}

std::vector<Impact> Simulator::independentImpacts(const std::vector<Impact>& impacts, int deform) const {
    std::vector<Impact> sorted = impacts;
    std::sort(sorted.begin(), sorted.end());
    
    std::unordered_set<Node*> nodes;
    std::vector<Impact> ans;
    for (const Impact& impact : sorted) {
        bool flag = true;
        for (int i = 0; i < 4; i++) {
            Node* node = impact.nodes[i];
            if ((deform == 1 || node->isFree) && nodes.find(node) != nodes.end()) {
                flag = false;
                break;
            }
        }

        if (flag) {
            ans.push_back(impact);
            for (int i = 0; i < 4; i++)
                nodes.insert(impact.nodes[i]);
        }
    }
    return ans;
}

thrust::device_vector<Impact> Simulator::independentImpacts(const thrust::device_vector<Impact>& impacts, int deform) const {
    int nImpacts = impacts.size();
    const Impact* impactsPointer = pointer(impacts);
    int nNodes = 4 * nImpacts;
    thrust::device_vector<Node*> nodes(nNodes), outputNodes(nNodes);
    Node** nodesPointer = pointer(nodes);
    Node** outputNodesPointer = pointer(outputNodes);
    thrust::device_vector<Pairfi> relativeImpacts(nNodes), outputRelativeImpacts(nNodes);
    Pairfi* relativeImpactsPointer = pointer(relativeImpacts);
    Pairfi* outputRelativeImpactsPointer = pointer(outputRelativeImpacts);
    thrust::device_vector<Impact> ans(nImpacts);
    Impact* ansPointer = pointer(ans);
    initializeImpactNodes<<<GRID_SIZE, BLOCK_SIZE>>>(nImpacts, impactsPointer, deform);
    CUDA_CHECK_LAST();

    int num, newNum = nImpacts;
    do {
        num = newNum;
        collectRelativeImpacts<<<GRID_SIZE, BLOCK_SIZE>>>(nImpacts, impactsPointer, deform, nodesPointer, relativeImpactsPointer);
        CUDA_CHECK_LAST();

        thrust::sort_by_key(nodes.begin(), nodes.end(), relativeImpacts.begin());
        auto iter = thrust::reduce_by_key(nodes.begin(), nodes.end(), relativeImpacts.begin(), outputNodes.begin(), outputRelativeImpacts.begin(), thrust::equal_to<Node*>(), thrust::minimum<Pairfi>());

        setImpactMinIndices<<<GRID_SIZE, BLOCK_SIZE>>>(iter.first - outputNodes.begin(), outputRelativeImpactsPointer, outputNodesPointer);
        CUDA_CHECK_LAST();

        checkIndependentImpacts<<<GRID_SIZE, BLOCK_SIZE>>>(nImpacts, impactsPointer, deform, ansPointer);
        CUDA_CHECK_LAST();

        newNum = thrust::count_if(ans.begin(), ans.end(), IsNull());
    } while (num > newNum);

    ans.erase(thrust::remove_if(ans.begin(), ans.end(), IsNull()), ans.end());
    CUDA_CHECK_LAST();

    return ans;
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

Vector3f Simulator::oldPosition(const Vector2f& u, const std::vector<BackupFace>& faces) const {
    for (const BackupFace& face : faces) {
        Vector3f b = face.barycentricCoordinates(u);
        if (b(0) >= -1e-6f && b(1) >= -1e-6f && b(2) >= -1e-5f)
            return face.position(b);
    }
}

Vector3f Simulator::oldPosition(const Face* face, const Vector3f& b, const std::vector<std::vector<BackupFace>>& faces) const {
    if (!face->isFree())
        return face->position(b);
    
    Vector2f u = b(0) * face->vertices[0]->u + b(1) * face->vertices[1]->u + b(2) * face->vertices[2]->u;
    for (int i = 0; i < cloths.size(); i++)
        if (cloths[i]->getMesh()->contain(face))
            return oldPosition(u, faces[i]);
}

void Simulator::checkIntersection(const Face* face0, const Face* face1, std::vector<Intersection>& intersections, const std::vector<std::vector<BackupFace>>& faces) const {
    Intersection intersection;
    Vector3f& b0 = intersection.b0;
    Vector3f& b1 = intersection.b1;
    if (checkIntersectionMidpoint(face0, face1, b0, b1)) {
        intersection.face0 = const_cast<Face*>(face0);
        intersection.face1 = const_cast<Face*>(face1);
        Vector3f x0 = oldPosition(face0, b0, faces);
        Vector3f x1 = oldPosition(face1, b1, faces);
        Vector3f& d = intersection.d;
        d = (x0 - x1).normalized();
        farthestPoint(face0, face1, d, b0, b1);
        intersections.push_back(intersection);
    }
}

thrust::device_vector<Intersection> Simulator::findIntersections(const std::vector<BVH*>& clothBvhs, const std::vector<BVH*>& obstacleBvhs, const std::vector<thrust::device_vector<BackupFace>>& faces) const {
    thrust::device_vector<Proximity> proximities = std::move(traverse(clothBvhs, obstacleBvhs, magic->collisionThickness));
    int nProximities = proximities.size();
    thrust::device_vector<Intersection> ans(nProximities);
    Intersection* ansPointer = pointer(ans);
    checkIntersectionsGpu<<<GRID_SIZE, BLOCK_SIZE>>>(nProximities, pointer(proximities), ansPointer);
    CUDA_CHECK_LAST();

    ans.erase(thrust::remove_if(ans.begin(), ans.end(), IsNull()), ans.end());
    int nAns = ans.size();
    thrust::device_vector<Vector3f> x(2 * nAns);
    Vector3f* xPointer = pointer(x);
    initializeOldPosition<<<GRID_SIZE, BLOCK_SIZE>>>(nAns, ansPointer, xPointer);
    CUDA_CHECK_LAST();

    for (int i = 0; i < cloths.size(); i++) {
        thrust::device_vector<Vertex*>& vertices = cloths[i]->getMesh()->getVerticesGpu();
        thrust::device_vector<int> indices(2 * nAns);
        int* indicesPointer = pointer(indices);
        thrust::device_vector<Vector2f> u(2 * nAns);
        Vector2f* uPointer = pointer(u);
        collectContainedFaces<<<GRID_SIZE, BLOCK_SIZE>>>(nAns, ansPointer, vertices.size(), pointer(vertices), indicesPointer, uPointer);
        CUDA_CHECK_LAST();

        u.erase(thrust::remove_if(u.begin(), u.end(), indices.begin(), IsNull()), u.end());
        indices.erase(thrust::remove(indices.begin(), indices.end(), -1), indices.end());
        computeOldPosition<<<GRID_SIZE, BLOCK_SIZE>>>(indices.size(), indicesPointer, uPointer, faces[i].size(), pointer(faces[i]), xPointer);
        CUDA_CHECK_LAST();
    }

    computeFarthestPoint<<<GRID_SIZE, BLOCK_SIZE>>>(nAns, xPointer, ansPointer);
    CUDA_CHECK_LAST();

    return ans;
}

void Simulator::resetObstacles() {
    for (Obstacle* obstacle : obstacles)
        obstacle->getMesh()->reset();
}

void Simulator::physicsStep() {
    for (Cloth* cloth : cloths)
        cloth->physicsStep(dt, magic->handleStiffness, gravity, wind);
    updateNodeGeometries();
    updateFaceGeometries();
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
                if (newImpacts.empty()) {
                    success = true;
                    break;
                }

                newImpacts = std::move(independentImpacts(newImpacts, deform));
                impacts.insert(impacts.end(), newImpacts.begin(), newImpacts.end());
                Optimization* optimization = new CollisionOptimization(impacts, magic->collisionThickness, deform, obstacleMass);
                optimization->solve();
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
        thrust::device_vector<Impact> impacts;
        for (int deform = 0; deform < 2; deform++) {
            impacts.clear();
            bool success = false;
            for (int i = 0; i < MAX_COLLISION_ITERATION; i++) {
                thrust::device_vector<Impact> newImpacts = std::move(findImpacts(clothBvhs, obstacleBvhs));
                if (newImpacts.empty()) {
                    success = true;
                    break;
                }

                newImpacts = std::move(independentImpacts(newImpacts, deform));
                impacts.insert(impacts.end(), newImpacts.begin(), newImpacts.end());
                Optimization* optimization = new CollisionOptimization(impacts, magic->collisionThickness, deform, obstacleMass);
                optimization->solve();
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
    }

    destroyBvhs(clothBvhs);
    destroyBvhs(obstacleBvhs);

    updateNodeGeometries();
    updateFaceGeometries();
    updateVelocities();
}

void Simulator::remeshingStep() {
    std::vector<BVH*> obstacleBvhs = std::move(buildObstacleBvhs(false));
    for (Cloth* cloth : cloths)
        cloth->remeshingStep(obstacleBvhs, 10.0f * magic->repulsionThickness);

    destroyBvhs(obstacleBvhs);

    updateStructures();
    updateNodeGeometries();
    updateFaceGeometries();
}

void Simulator::separationStep(const std::vector<std::vector<BackupFace>>& faces) {
    std::vector<BVH*> clothBvhs = std::move(buildClothBvhs(false));
    std::vector<BVH*> obstacleBvhs = std::move(buildObstacleBvhs(false));
    float obstacleArea = 1e3f;

    std::vector<Intersection> intersections;
    for (int deform = 0; deform < 2; deform++) {
        intersections.clear();
        bool success = false;
        for (int i = 0; i < MAX_SEPARATION_ITERATION; i++) {
            if (!intersections.empty())
                updateActive(clothBvhs, obstacleBvhs, intersections);
            
            std::vector<Intersection> newIntersections;
            traverse(clothBvhs, obstacleBvhs, magic->collisionThickness, [&](const Face* face0, const Face* face1, float thickness) {
                checkIntersection(face0, face1, newIntersections, faces);
            });
            if (newIntersections.empty()) {
                success = true;
                break;
            }

            intersections.insert(intersections.end(), newIntersections.begin(), newIntersections.end());
            Optimization* optimization = new SeparationOptimization(intersections, magic->collisionThickness, deform, obstacleArea);
            optimization->solve();
            delete optimization;

            updateFaceGeometries();
            updateBvhs(clothBvhs);
            if (deform == 1) {
                updateBvhs(obstacleBvhs);
                obstacleArea *= 0.5f;
            }
        }
        if (success)
            break;
    }

    destroyBvhs(clothBvhs);
    destroyBvhs(obstacleBvhs);

    updateNodeGeometries();
    updateVelocities();
}

void Simulator::separationStep(const std::vector<thrust::device_vector<BackupFace>>& faces) {
    std::vector<BVH*> clothBvhs = std::move(buildClothBvhs(false));
    std::vector<BVH*> obstacleBvhs = std::move(buildObstacleBvhs(false));
    float obstacleArea = 1e3f;

    thrust::device_vector<Intersection> intersections;
    for (int deform = 0; deform < 2; deform++) {
        intersections.clear();
        bool success = false;
        for (int i = 0; i < MAX_SEPARATION_ITERATION; i++) {
            thrust::device_vector<Intersection> newIntersections = std::move(findIntersections(clothBvhs, obstacleBvhs, faces));
            if (newIntersections.empty()) {
                success = true;
                break;
            }

            intersections.insert(intersections.end(), newIntersections.begin(), newIntersections.end());
            Optimization* optimization = new SeparationOptimization(intersections, magic->collisionThickness, deform, obstacleArea);
            optimization->solve();
            delete optimization;

            updateFaceGeometries();
            updateBvhs(clothBvhs);
            if (deform == 1) {
                updateBvhs(obstacleBvhs);
                obstacleArea *= 0.5f;
            }
        }
        if (success)
            break;
    }

    destroyBvhs(clothBvhs);
    destroyBvhs(obstacleBvhs);

    updateNodeGeometries();
    updateVelocities();
}

void Simulator::updateStructures() {
    for (Cloth* cloth : cloths)
        cloth->getMesh()->updateStructures();
}

void Simulator::updateNodeGeometries() {
    for (Cloth* cloth : cloths)
        cloth->getMesh()->updateNodeGeometries();
}

void Simulator::updateFaceGeometries() {
    for (Cloth* cloth : cloths)
        cloth->getMesh()->updateFaceGeometries();
}

void Simulator::updateVelocities() {
    for (Cloth* cloth : cloths)
        cloth->getMesh()->updateVelocities(dt);
}

void Simulator::updateRenderingData(bool rebind) {
    for (Cloth* cloth : cloths)
        cloth->getMesh()->updateRenderingData(rebind);
}

void Simulator::step(bool offline) {
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
        nFrames++;
        if (!gpu) {
            std::vector<std::vector<BackupFace>> faces(cloths.size());
            for (int i = 0; i < cloths.size(); i++)
                faces[i] = std::move(cloths[i]->getMesh()->backupFaces());

            remeshingStep();
            auto t3 = std::chrono::high_resolution_clock::now();
            d = t3 - t2;
            std::cout << ", Remeshing Step: " << d.count() << "s";

            separationStep(faces);
            auto t4 = std::chrono::high_resolution_clock::now();
            d = t4 - t3;
            std::cout << ", Separation Step: " << d.count() << "s";
        } else {
            std::vector<thrust::device_vector<BackupFace>> faces(cloths.size());
            for (int i = 0; i < cloths.size(); i++)
                faces[i] = std::move(cloths[i]->getMesh()->backupFacesGpu());

            remeshingStep();
            auto t3 = std::chrono::high_resolution_clock::now();
            d = t3 - t2;
            std::cout << ", Remeshing Step: " << d.count() << "s";

            separationStep(faces);
            auto t4 = std::chrono::high_resolution_clock::now();
            d = t4 - t3;
            std::cout << ", Separation Step: " << d.count() << "s";
        }

        if (!offline)
            updateRenderingData(true);
        else
            save();
    } else if (!offline)
        updateRenderingData(false);

    std::cout << std::endl;
}

void Simulator::bind() {
    for (Cloth* cloth : cloths)
        cloth->bind();
    for (Obstacle* obstacle : obstacles)
        obstacle->bind();
}

void Simulator::render() const {
    Vector3f lightDirection = renderer->getLightDirection();
    Vector3f cameraPosition = renderer->getCameraPosition();
    Matrix4x4f model = renderer->getModel();
    Matrix4x4f view = renderer->getView();
    Matrix4x4f projection = renderer->getProjection();

    for (const Cloth* cloth : cloths)
        cloth->render(model, view, projection, cameraPosition, lightDirection);
    
    for (const Obstacle* obstacle : obstacles)
        obstacle->render(model, view, projection, cameraPosition, lightDirection);
}

void Simulator::load() {
    for (int i = 0; i < cloths.size(); i++) {
        std::string path = directory + "/frame" + std::to_string(nFrames) + "_cloth" + std::to_string(i) + ".obj";
        if (std::filesystem::exists(path))
            cloths[i]->load(path);
    }
    nFrames++;
}

void Simulator::save() {
    for (int i = 0; i < cloths.size(); i++)
        cloths[i]->save(directory + "/frame" + std::to_string(nFrames) + "_cloth" + std::to_string(i) + ".obj", json["cloths"][i]);
    
    for (int i = 0; i < obstacles.size(); i++)
        obstacles[i]->save(directory + "/frame" + std::to_string(nFrames) + "_obstacle" + std::to_string(i) + ".obj", json["obstacles"][i]);

    std::ofstream fout(directory + "/config.json");
    fout << json;
}

void Simulator::findLastFrame() {
    bool flag;
    do {
        nFrames++;
        flag = true;
        for (int i = 0; i < cloths.size(); i++)
            if (!std::filesystem::exists(directory + "/frame" + std::to_string(nFrames) + "_cloth" + std::to_string(i) + ".obj")) {
                flag = false;
                break;
            }
    } while (flag);

    nFrames--;
    nSteps = nFrames * frameSteps;
}

void Simulator::simulate() {
    if (!gpu) {
        std::vector<std::vector<BackupFace>> faces(cloths.size());
        for (int i = 0; i < cloths.size(); i++)
            faces[i] = std::move(cloths[i]->getMesh()->backupFaces());

        remeshingStep();
        separationStep(faces);
    } else {
        std::vector<thrust::device_vector<BackupFace>> faces(cloths.size());
        for (int i = 0; i < cloths.size(); i++)
            faces[i] = std::move(cloths[i]->getMesh()->backupFacesGpu());

        remeshingStep();
        separationStep(faces);
    }
    bind();

    while (!glfwWindowShouldClose(renderer->getWindow())) {
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        render();
        if (!renderer->getPause())
            step(false);

        glfwSwapBuffers(renderer->getWindow());
        glfwPollEvents();
    }
}

void Simulator::simulateOffline() {
    if (!std::filesystem::is_directory(directory))
        std::filesystem::create_directories(directory);

    if (!gpu) {
        std::vector<std::vector<BackupFace>> faces(cloths.size());
        for (int i = 0; i < cloths.size(); i++)
            faces[i] = std::move(cloths[i]->getMesh()->backupFaces());

        remeshingStep();
        separationStep(faces);
    } else {
        std::vector<thrust::device_vector<BackupFace>> faces(cloths.size());
        for (int i = 0; i < cloths.size(); i++)
            faces[i] = std::move(cloths[i]->getMesh()->backupFacesGpu());

        remeshingStep();
        separationStep(faces);
    }
    save();

    while (true)
        step(true);
}

void Simulator::resume() {
    findLastFrame();
    bind();
    
    while (!glfwWindowShouldClose(renderer->getWindow())) {
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        render();
        if (!renderer->getPause())
            step(false);

        glfwSwapBuffers(renderer->getWindow());
        glfwPollEvents();
    }
}

void Simulator::resumeOffline() {
    findLastFrame();

    while (true)
        step(true);
}

void Simulator::replay() {
    load();
    bind();

    while (!glfwWindowShouldClose(renderer->getWindow())) {
        auto t0 = std::chrono::high_resolution_clock::now();
       
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        render();
        if (!renderer->getPause()) {
            load();
            updateRenderingData(true);
        }

        glfwSwapBuffers(renderer->getWindow());
        glfwPollEvents();

        std::chrono::duration<float> d;
        do {
            auto t1 = std::chrono::high_resolution_clock::now();
            d = t1 - t0;
        } while (d.count() < frameTime);
    }
}

void Simulator::start() {
    switch (mode) {
    case Simulate:
        simulate();
        break;
    case SimulateOffline:
        simulateOffline();
        break;
    case Resume:
        resume();
        break;
    case ResumeOffline:
        resumeOffline();
        break;
    case Replay:
        replay();
        break;
    }
}