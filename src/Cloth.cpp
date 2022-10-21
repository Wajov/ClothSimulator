#include "Cloth.hpp"

Cloth::Cloth(const Json::Value& json) {
    Transform* transform = new Transform(json["transform"]);
    material = new Material(json["materials"]);
    mesh = new Mesh(json["mesh"], transform, material);

    std::vector<Vertex>& vertices = mesh->getVertices();
    for (const Json::Value& handleJson : json["handles"])
        for (const Json::Value& nodeJson : handleJson["nodes"]) {
            int index = parseInt(nodeJson);
            handles.push_back(new Handle(&vertices[index], vertices[index].x));
        }

    remeshing = new Remeshing(json["remeshing"]);

    edgeShader = new Shader("shader/VertexShader.glsl", "shader/EdgeFragmentShader.glsl");
    faceShader = new Shader("shader/VertexShader.glsl", "shader/FaceFragmentShader.glsl");
    delete transform;
}

Cloth::~Cloth() {
    delete mesh;
    delete material;
    for (const Handle* handle : handles)
        delete handle;
    delete edgeShader;
    delete faceShader;
}

Vector3i Cloth::indices(const Vertex* v0, const Vertex* v1, const Vertex* v2) const {
    Vector3i ans;
    ans(0) = v0->index;
    ans(1) = v1->index;
    ans(2) = v2->index;
    return ans;
}

Vector4i Cloth::indices(const Vertex* v0, const Vertex* v1, const Vertex* v2, const Vertex* v3) const {
    Vector4i ans;
    ans(0) = v0->index;
    ans(1) = v1->index;
    ans(2) = v2->index;
    ans(3) = v3->index;
    return ans;
}

void Cloth::addSubMatrix(const MatrixXxXf& B, const VectorXi& indices, Eigen::SparseMatrix<float>& A) const {
    for (int i = 0; i < indices.rows(); i++) {
        int x = indices[i];
        for (int j = 0; j < indices.rows(); j++) {
            int y = indices[j];
            for (int k = 0; k < 3; k++)
                for (int h = 0; h < 3; h++)
                    A.coeffRef(3 * x + k, 3 * y + h) += B(3 * i + k, 3 * j + h);
        }
    }
}

void Cloth::addSubVector(const VectorXf& b, const VectorXi& indices, VectorXf& a) const {
    for (int i = 0; i < indices.rows(); i++) {
        int x = indices[i];
        for (int j = 0; j < 3; j++)
            a(3 * x + j) += b(3 * i + j);
    }
}

float Cloth::distance(const Vector3f& x, const Vector3f& a, const Vector3f& b) const {
    Vector3f e = b - a;
    Vector3f t = x - a;
    Vector3f r = e * e.dot(t) / e.squaredNorm();
    return (t - r).norm();
}

Vector2f Cloth::barycentricWeights(const Vector3f& x, const Vector3f& a, const Vector3f& b) const {
    Vector3f e = b - a;
    float t = e.dot(x - a) / e.squaredNorm();
    return Vector2f(1.0f - t, t);
}

std::pair<Vector9f, Matrix9x9f> Cloth::stretchingForce(const Face* face) const {
    Matrix3x2f F = face->derivative(face->getVertex(0)->x, face->getVertex(1)->x, face->getVertex(2)->x);
    Matrix2x2f G = 0.5f * (F.transpose() * F - Matrix2x2f::Identity());

    Matrix2x2f Y = face->getInverse();
    Matrix2x3f D = concatenateToMatrix(-Y.row(0) - Y.row(1), Y.row(0), Y.row(1));
    Matrix3x9f Du = kronecker(D.row(0), Matrix3x3f::Identity());
    Matrix3x9f Dv = kronecker(D.row(1), Matrix3x3f::Identity());

    Vector3f fu = F.col(0);
    Vector3f fv = F.col(1);

    Vector9f fuu = Du.transpose() * fu;
    Vector9f fvv = Dv.transpose() * fv;
    Vector9f fuv = 0.5f * (Du.transpose() * fv + Dv.transpose() * fu);

    Vector4f k = material->stretchingStiffness(G);

    Vector9f grad = k(0) * G(0, 0) * fuu + k(2) * G(1, 1) * fvv + k(1) * (G(0, 0) * fvv + G(1, 1) * fuu) + 2.0f * k(3) * G(0, 1) * fuv;
    Matrix9x9f hess = k(0) * (fuu * fuu.transpose() + std::max(G(0, 0), 0.0f) * Du.transpose() * Du)
                    + k(2) * (fvv * fvv.transpose() + std::max(G(1, 1), 0.0f) * Dv.transpose() * Dv)
                    + k(1) * (fuu * fvv.transpose() + std::max(G(0, 0), 0.0f) * Dv.transpose() * Dv + fvv * fuu.transpose() + std::max(G(1, 1), 0.0f) * Du.transpose() * Du)
                    + 2.0f * k(3) * fuv * fuv.transpose();

    float area = face->getArea();
    return std::make_pair(-area * grad, -area * hess);
}

std::pair<Vector12f, Matrix12x12f> Cloth::bendingForce(const Edge* edge) const {
    Vector3f x0 = edge->getVertex(0)->x;
    Vector3f x1 = edge->getVertex(1)->x;
    const std::vector<Vertex*>& opposites = edge->getOpposites();
    Vector3f x2 = opposites[0]->x;
    Vector3f x3 = opposites[1]->x;
    const std::vector<Face*>& adjacents = edge->getAdjacents();
    Vector3f n0 = adjacents[0]->getNormal();
    Vector3f n1 = adjacents[1]->getNormal();
    float length = edge->getLength();
    float angle = edge->getAngle();
    float area = adjacents[0]->getArea() + adjacents[1]->getArea();

    float h0 = distance(x2, x0, x1);
    float h1 = distance(x3, x0, x1);
    Vector2f w0 = barycentricWeights(x2, x0, x1);
    Vector2f w1 = barycentricWeights(x3, x0, x1);

    Vector12f dtheta = concatenateToVector(-w0(0) * n0 / h0 - w1(0) * n1 / h1, -w0(1) * n0 / h0 - w1(1) * n1 / h1, n0 / h0, n1 / h1);

    float k = material->bendingStiffness(length, angle, area, edge->getVertex(1)->u - edge->getVertex(0)->u);
    float coefficient = -0.25f * k * length * length / area;

    return std::make_pair(coefficient * angle * dtheta, coefficient * dtheta * dtheta.transpose());
}

void Cloth::init(Eigen::SparseMatrix<float>& A, VectorXf& b) const {
    std::vector<Vertex>& vertices = mesh->getVertices();
    int n = vertices.size();

    A.resize(3 * n, 3 * n);
    A.setZero();
    for (int i = 0; i < n; i++) {
        float m = vertices[i].m;
        A.coeffRef(3 * i, 3 * i) += m;
        A.coeffRef(3 * i + 1, 3 * i + 1) += m;
        A.coeffRef(3 * i + 2, 3 * i + 2) += m;
    }

    b.resize(3 * n);
    b.setZero();
}

void Cloth::addExternalForces(float dt, const Vector3f& gravity, const Wind* wind, Eigen::SparseMatrix<float>& A, VectorXf& b) const {
    std::vector<Vertex>& vertices = mesh->getVertices();
    for (const Vertex& vertex : vertices)
        b.block<3, 1>(3 * vertex.index, 0) += dt * vertex.m * gravity;

    std::vector<Face*>& faces = mesh->getFaces();
    for (const Face* face : faces) {
        float area = face->getArea();
        Vector3f normal = face->getNormal();
        Vector3f average = (face->getVertex(0)->v + face->getVertex(1)->v + face->getVertex(2)->v) / 3.0f;
        Vector3f relative = wind->getVelocity() - average;
        float vn = normal.dot(relative);
        Vector3f vt = relative - vn * normal;
        Vector3f force = area * (wind->getDensity() * std::abs(vn) * vn * normal + wind->getDrag() * vt) / 3.0f;
        b.block<3, 1>(3 * face->getVertex(0)->index, 0) += dt * force;
        b.block<3, 1>(3 * face->getVertex(1)->index, 0) += dt * force;
        b.block<3, 1>(3 * face->getVertex(2)->index, 0) += dt * force;
    }
}

void Cloth::addInternalForces(float dt, Eigen::SparseMatrix<float>& A, VectorXf& b) const {
    std::vector<Face*>& faces = mesh->getFaces();
    for (const Face* face : faces) {
        Vertex* v0 = face->getVertex(0);
        Vertex* v1 = face->getVertex(1);
        Vertex* v2 = face->getVertex(2);
        Vector9f v = concatenateToVector(v0->v, v1->v, v2->v);

        std::pair<Vector9f, Matrix9x9f> pair = stretchingForce(face);
        Vector9f f = pair.first;
        Matrix9x9f J = pair.second;

        Vector3i vertexIndices = indices(v0, v1, v2);
        addSubMatrix(-dt * dt * J, vertexIndices, A);
        addSubVector(dt * (f + dt * J * v), vertexIndices, b);
    }

    std::vector<Edge*>& edges = mesh->getEdges();
    for (const Edge* edge : edges) {
        const std::vector<Vertex*>& opposites = edge->getOpposites();
        if (opposites.size() == 2) {
            Vertex* v0 = edge->getVertex(0);
            Vertex* v1 = edge->getVertex(1);
            Vertex* v2 = opposites[0];
            Vertex* v3 = opposites[1];
            Vector12f v = concatenateToVector(v0->v, v1->v, v2->v, v3->v);

            std::pair<Vector12f, Matrix12x12f> pair = bendingForce(edge);
            Vector12f f = pair.first;
            Matrix12x12f J = pair.second;

            Vector4i vertexIndices = indices(v0, v1, v2, v3);
            addSubMatrix(-dt * dt * J, vertexIndices, A);
            addSubVector(dt * (f + dt * J * v), vertexIndices, b);
        }
    }
}

void Cloth::addHandleForces(float dt, float stiffness, Eigen::SparseMatrix<float>& A, VectorXf& b) const {
    for (const Handle* handle : handles) {
        Vertex* vertex = handle->getVertex();
        Vector3f position = handle->getPosition();
        int index = vertex->index;
        A.coeffRef(3 * index, 3 * index) += dt * dt * stiffness;
        A.coeffRef(3 * index + 1, 3 * index + 1) += dt * dt * stiffness;
        A.coeffRef(3 * index + 2, 3 * index + 2) += dt * dt * stiffness;
        b.block<3, 1>(3 * index, 0) += dt * ((position - vertex->x) - dt * vertex->v) * stiffness;
    }
    // for (const Constraint* constraint : constraints) {
    //     std::vector<Gradient*> gradient = constraint->energyGradient();
    //     std::vector<Hessian*> hessian = constraint->energyHessian();
    //     for (const Gradient* grad : gradient)
    //         b.block<3, 1>(3 * grad->getIndex(), 0) -= dt * grad->getValue();
    //     for (const Hessian* hess : hessian) {
    //         A.block<
    //     }
    // }
}

Matrix2x2f Cloth::compressionMetric(const Matrix2x2f& G, const Matrix2x2f& S2, float refine, float ribStiffening) const {
    Matrix2x2f P;
    P << S2(1, 1), -S2(1, 0), -S2(0, 1), S2(0, 0);
    Matrix2x2f D = G.transpose() * G - 4.0f * sqr(refine) * P * ribStiffening;
    return max(-G + sqrt(D), 0.0f) / (2.0f * sqr(refine));
}

Matrix2x2f Cloth::obstacleMetric(const Face* face, const std::vector<Plane>& planes) const {
    Matrix2x2f ans = Matrix2x2f::Zero();
    for (int i = 0; i < 3; i++) {
        Plane plane = planes[face->getVertex(i)->index];
        if (plane.n.squaredNorm() == 0.0f)
            continue;
        float h[3];
        for (int j = 0; j < 3; j++)
            h[j] = (face->getVertex(j)->x - plane.p).dot(plane.n);
        Vector2f dh = face->getInverse().transpose() * Vector2f(h[1] - h[0], h[2] - h[0]);
        ans += dh * dh.transpose() / sqr(h[i]);
    }
    return ans / 3.0f;
}

Matrix2x2f Cloth::maxTensor(const Matrix2x2f M[]) const {
    int n = 0;
    Disk d[5];
    for (int i = 0; i < 5; i++)
        if (M[i].trace() != 0.0f) {
            d[n].o = Vector2f(0.5f * (M[i](0, 0) - M[i](1, 1)), 0.5f * (M[i](0, 1) + M[i](1, 0)));
            d[n].r = 0.5f * (M[i](0, 0) + M[i](1, 1));
            n++;
        }

    Disk disk;
    disk = d[0];
    for (int i = 1; i < n; i++)
        if (!disk.enclose(d[i])) {
            disk = d[i];
            for (int j = 0; j < i; j++)
                if (!disk.enclose(d[j])) {
                    disk = Disk::circumscribedDisk(d[i], d[j]);
                    for (int k = 0; k < j; k++)
                        if (!disk.enclose(d[k]))
                            disk = Disk::circumscribedDisk(d[i], d[j], d[k]);
                }
        }

    Matrix2x2f ans;
    ans(0, 0) = disk.r + disk.o(0);
    ans(0, 1) = ans(1, 0) = disk.o(1);
    ans(1, 1) = disk.r - disk.o(0);
    return ans;
}

Matrix2x2f Cloth::faceSizing(const Face* face, const std::vector<Plane>& planes, float ribStiffening) const {
    Vertex* v0 = face->getVertex(0);
    Vertex* v1 = face->getVertex(1);
    Vertex* v2 = face->getVertex(2);
    Matrix2x2f M[5];

    Matrix2x2f Sw1 = face->curvature();
    M[0] = (Sw1.transpose() * Sw1) / sqr(remeshing->refineAngle);
    Matrix3x2f Sw2 = face->derivative(v0->n, v1->n, v2->n);
    M[1] = (Sw2.transpose() * Sw2) / sqr(remeshing->refineAngle);
    Matrix3x2f V = face->derivative(v0->v, v1->v, v2->v);
    M[2] = (V.transpose() * V) / sqr(remeshing->refineVelocity);
    Matrix3x2f F = face->derivative(v0->x, v1->x, v2->x);
    M[3] = compressionMetric(F.transpose() * F - Matrix2x2f::Identity(), Sw2.transpose() * Sw2, remeshing->refineCompression, ribStiffening);
    M[4] = obstacleMetric(face, planes);
    Matrix2x2f S = maxTensor(M);

    Matrix2x2f Q;
    Vector2f l;
    eigenvalueDecomposition(S, Q, l);
    for (int i = 0; i < 2; i++)
        l(i) = std::clamp(l(i), 1.0f / sqr(remeshing->sizeMax), 1.0f / sqr(remeshing->sizeMin));
    float lMax = std::max(l(0), l(1));
    float lMin = lMax * sqr(remeshing->aspectMin);
    for (int i = 0; i < 2; i++)
        l(i) = std::max(l(i), lMin);
    return Q * diagonal(l) * Q.transpose();
}

Mesh* Cloth::getMesh() const {
    return mesh;
}

void Cloth::readDataFromFile(const std::string& path) {
    mesh->readDataFromFile(path);
    mesh->updateGeometry(material);
}

void Cloth::physicsStep(float dt, float handleStiffness, const Vector3f& gravity, const Wind* wind) {
    Eigen::SparseMatrix<float> A;
    VectorXf b;

    init(A, b);
    addExternalForces(dt, gravity, wind, A, b);
    addInternalForces(dt, A, b);
    addHandleForces(dt, handleStiffness, A, b);

    Eigen::SimplicialLLT<Eigen::SparseMatrix<float>> cholesky;
    cholesky.compute(A);
    VectorXf dv = cholesky.solve(b);

    std::vector<Vertex>& vertices = mesh->getVertices();
    for (int i = 0; i < vertices.size(); i++) {
        vertices[i].x0 = vertices[i].x;
        vertices[i].v += dv.block<3, 1>(3 * i, 0);
        vertices[i].x += vertices[i].v * dt;
    }
}

void Cloth::remeshingStep(const std::vector<BVH*>& obstacleBvhs, float thickness, float ribStiffening) {
    std::vector<Vertex>& vertices = mesh->getVertices();
    std::vector<Plane> planes(vertices.size(), Plane(Vector3f(0.0f, 0.0f, 0.0f), Vector3f(0.0f, 0.0f, 0.0f)));
    for (int i = 0; i < vertices.size(); i++) {
        NearPoint point(thickness, vertices[i].x);
        for (const BVH* obstacleBvh : obstacleBvhs)
            obstacleBvh->findNearestPoint(vertices[i].x, point);
    
        if (point.x != vertices[i].x) {
            planes[i].p = point.x;
            planes[i].n = (vertices[i].x - point.x).normalized();
        }
    }

    for (Vertex& vertex : vertices) {
        vertex.a = 0.0f;
        vertex.sizing = Matrix2x2f::Zero();
    }

    std::vector<Face*>& faces = mesh->getFaces();
    for (Face* face : faces) {
        float area = face->getArea();
        Matrix2x2f sizing = faceSizing(face, planes, ribStiffening);
        for (int i = 0; i < 3; i++) {
            Vertex* vertex = face->getVertex(i);
            vertex->a += area;
            vertex->sizing += area * sizing;
        }
    }

    for (Vertex& vertex : vertices)
        vertex.sizing /= vertex.a;

    std::ofstream fout("output_sizing.txt");
    fout.precision(20);
    for (const Vertex& vertex : vertices) {
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                fout << vertex.sizing(i, j) << ' ';
        fout << std::endl;
    }

    // TODO
}

void Cloth::updateGeometry() {
    mesh->updateGeometry(material);
}

void Cloth::updateVelocity(float dt) {
    mesh->updateVelocity(dt);
}

void Cloth::updateRenderingData() const {
    mesh->updateRenderingData();
}

void Cloth::render(const Matrix4x4f& model, const Matrix4x4f& view, const Matrix4x4f& projection, const Vector3f& cameraPosition, const Vector3f& lightPosition, float lightPower) const {
    edgeShader->use();
    edgeShader->setMat4("model", model);
    edgeShader->setMat4("view", view);
    edgeShader->setMat4("projection", projection);
    mesh->renderEdge();

    faceShader->use();
    faceShader->setMat4("model", model);
    faceShader->setMat4("view", view);
    faceShader->setMat4("projection", projection);
    faceShader->setVec3("color", Vector3f(0.6f, 0.7f, 1.0f));
    faceShader->setVec3("cameraPosition", cameraPosition);
    faceShader->setVec3("lightPosition", lightPosition);
    faceShader->setFloat("lightPower", lightPower);
    mesh->renderFace();
}
