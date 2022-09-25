#include "Cloth.hpp"

Cloth::Cloth(const Json::Value& json) {
    Vector3f translate;
    for (int i = 0; i < 3; i++)
        translate(i) = json["transform"]["translate"][i].asDouble();
    mesh = new Mesh(json["mesh"], translate);

    for (int i = 0; i < json["materials"].size(); i++)
        materials.push_back(new Material(json["materials"][i]));
}

Cloth::~Cloth() {
    for (const Material* material : materials)
        delete material;
    for (const Handle* handle : handles)
        delete handle;
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

std::pair<Vector9f, Matrix9x9f> Cloth::stretchingForce(const Face* face, const Material* material) const {
    Vector3f x0 = face->getV0()->position;
    Vector3f x1 = face->getV1()->position;
    Vector3f x2 = face->getV2()->position;
    Vector3f d1 = x1 - x0;
    Vector3f d2 = x2 - x0;
    Vector3f n = face->getNormal();

    Matrix3x3f F = concatenateToMatrix(d1, d2, n) * face->getInverse();
    Matrix3x3f G = 0.5f * (F.transpose() * F - Matrix3x3f::Identity());

    Matrix3x3f Y = face->getInverse();
    Matrix3x3f D = concatenateToMatrix(-Y.row(0) - Y.row(1), Y.row(0), Y.row(1)).transpose();
    Matrix3x9f Du = kronecker(D.col(0).transpose(), Matrix3x3f::Identity());
    Matrix3x9f Dv = kronecker(D.col(1).transpose(), Matrix3x3f::Identity());

    Vector9f x = concatenateToVector(x0, x1, x2);

    Vector3f fu = Du * x;
    Vector3f fv = Dv * x;

    Vector9f fuu = Du.transpose() * fu;
    Vector9f fvv = Dv.transpose() * fv;
    Vector9f fuv = 0.5f * (Du.transpose() * fv + Dv.transpose() * fu);

    Vector4f k = material->stretchingStiffness(G.block<2, 2>(0, 0));

    Vector9f grad = k(0) * G(0, 0) * fuu + k(2) * G(1, 1) * fvv + k(1) * (G(0, 0) * fvv + G(1, 1) * fuu) + 2.0f * k(3) * G(0, 1) * fuv;
    Matrix9x9f hess = k(0) * (fuu * fuu.transpose() + std::max(G(0, 0), 0.0f) * Du.transpose() * Du)
                    + k(2) * (fvv * fvv.transpose() + std::max(G(1, 1), 0.0f) * Dv.transpose() * Dv)
                    + k(1) * (fuu * fvv.transpose() + std::max(G(0, 0), 0.0f) * Dv.transpose() * Dv + fvv * fuu.transpose() + std::max(G(1, 1), 0.0f) * Du.transpose() * Du)
                    + 2.0f * k(3) * fuv * fuv.transpose();

    float area = face->getArea();
    return std::make_pair(-area * grad, -area * hess);
}

std::pair<Vector12f, Matrix12x12f> Cloth::bendingForce(const Edge* edge, const Material* material) const {
    Vector3f x0 = edge->getV0()->position;
    Vector3f x1 = edge->getV1()->position;
    const std::vector<Vertex*>& opposites = edge->getOpposites();
    Vector3f x2 = opposites[0]->position;
    Vector3f x3 = opposites[1]->position;
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

    float k = material->bendingStiffness(length, angle, area, edge->getV1()->uv - edge->getV0()->uv);
    float coefficient = -0.25f * k * length * length / area;

    return std::make_pair(coefficient * angle * dtheta, coefficient * dtheta * dtheta.transpose());
}

std::vector<Constraint*> Cloth::getConstraints() const {
    std::vector<Constraint*> ans;
    for (const Handle* handle : handles)
        ans.push_back(handle->getConstraint());
    return ans;
}

void Cloth::init(Eigen::SparseMatrix<float>& A, VectorXf& b) const {
    const std::vector<Vertex>& vertices = mesh->getVertices();
    int n = vertices.size();

    A.resize(3 * n, 3 * n);
    A.setZero();
    for (int i = 0; i < n; i++) {
        float mass = vertices[i].mass;
        A.coeffRef(3 * i, 3 * i) += mass;
        A.coeffRef(3 * i + 1, 3 * i + 1) += mass;
        A.coeffRef(3 * i + 2, 3 * i + 2) += mass;
    }

    b.resize(3 * n);
    b.setZero();
}

void Cloth::addExternalForces(float dt, const Vector3f& gravity, const Wind* wind, Eigen::SparseMatrix<float>& A, VectorXf& b) const {
    const std::vector<Vertex>& vertices = mesh->getVertices();
    for (const Vertex& vertex : vertices)
        b.block<3, 1>(3 * vertex.index, 0) += dt * vertex.mass * gravity;

    const std::vector<Face*>& faces = mesh->getFaces();
    for (const Face* face : faces) {
        float area = face->getArea();
        Vector3f normal = face->getNormal();
        Vector3f average = (face->getV0()->velocity + face->getV1()->velocity + face->getV2()->velocity) / 3.0f;
        Vector3f relative = wind->getVelocity() - average;
        float vn = normal.dot(relative);
        Vector3f vt = relative - vn * normal;
        Vector3f force = area * (wind->getDensity() * std::abs(vn) * vn * normal + wind->getDrag() * vt) / 3.0f;
        b.block<3, 1>(3 * face->getV0()->index, 0) += dt * force;
        b.block<3, 1>(3 * face->getV1()->index, 0) += dt * force;
        b.block<3, 1>(3 * face->getV2()->index, 0) += dt * force;
    }
}

void Cloth::addInternalForces(float dt, const Material* material, Eigen::SparseMatrix<float>& A, VectorXf& b) const {
    const std::vector<Face*>& faces = mesh->getFaces();
    for (const Face* face : faces) {
        Vertex* v0 = face->getV0();
        Vertex* v1 = face->getV1();
        Vertex* v2 = face->getV2();
        Vector9f v = concatenateToVector(v0->velocity, v1->velocity, v2->velocity);

        std::pair<Vector9f, Matrix9x9f> pair = stretchingForce(face, material);
        Vector9f f = pair.first;
        Matrix9x9f J = pair.second;

        Vector3i vertexIndices = indices(v0, v1, v2);
        addSubMatrix(-dt * dt * J, vertexIndices, A);
        addSubVector(dt * (f + dt * J * v), vertexIndices, b);
    }

    const std::vector<Edge*>& edges = mesh->getEdges();
    for (const Edge* edge : edges) {
        const std::vector<Vertex*>& opposites = edge->getOpposites();
        if (opposites.size() == 2) {
            Vertex* v0 = edge->getV0();
            Vertex* v1 = edge->getV1();
            Vertex* v2 = opposites[0];
            Vertex* v3 = opposites[1];
            Vector12f v = concatenateToVector(v0->velocity, v1->velocity, v2->velocity, v3->velocity);

            std::pair<Vector12f, Matrix12x12f> pair = bendingForce(edge, material);
            Vector12f f = pair.first;
            Matrix12x12f J = pair.second;

            Vector4i vertexIndices = indices(v0, v1, v2, v3);
            addSubMatrix(-dt * dt * J, vertexIndices, A);
            addSubVector(dt * (f + dt * J * v), vertexIndices, b);
        }
    }
}

void Cloth::addConstraintForces(float dt, const std::vector<Constraint*>& constraints, Eigen::SparseMatrix<float>& A, VectorXf& b) const {
    for (const Handle* handle : handles) {
        Vertex* vertex = handle->getVertex();
        Vector3f position = handle->getPosition();
        int index = vertex->index;
        A.coeffRef(3 * index, 3 * index) += dt * dt * 1000.0f;
        A.coeffRef(3 * index + 1, 3 * index + 1) += dt * dt * 1000.0f;
        A.coeffRef(3 * index + 2, 3 * index + 2) += dt * dt * 1000.0f;
        b.block<3, 1>(3 * index, 0) += dt * ((position - vertex->position) - dt * vertex->velocity) * 1000.0f;
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

Mesh* Cloth::getMesh() const {
    return mesh;
}

void Cloth::addHandle(int index) {
    const std::vector<Vertex>& vertices = mesh->getVertices();
    handles.push_back(new Handle(&vertices[index], vertices[index].position));
}

void Cloth::readDataFromFile(const std::string& path) {
    mesh->readDataFromFile(path);
}

void Cloth::update(float dt, const Vector3f& gravity, const Wind* wind) {
    mesh->updateData(materials[0]);

    Eigen::SparseMatrix<float> A;
    VectorXf b;

    std::vector<Constraint*> constraints = getConstraints();
    init(A, b);
    addExternalForces(dt, gravity, wind, A, b);
    addInternalForces(dt, materials[0], A, b);
    addConstraintForces(dt, constraints, A, b);

    Eigen::SimplicialLLT<Eigen::SparseMatrix<float>> cholesky;
    cholesky.compute(A);
    VectorXf dv = cholesky.solve(b);

    mesh->update(dt, dv);

    std::ofstream fout("output.txt");
    fout.precision(20);
    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < A.cols(); j++)
            fout << A.coeff(i, j) << ' ';
        fout << std::endl;
    }
    for (int i = 0; i < b.rows(); i++)
        fout << b(i) << ' ' ;
    fout << std::endl;
    fout.close();

    exit(0);
}

void Cloth::renderEdge() const {
    mesh->renderEdge();
}

void Cloth::renderFace() const {
    mesh->renderFace();
}
