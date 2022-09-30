#include "Material.hpp"

Material::Material(const Json::Value& json) {
    thicken = parseFloat(json["thicken"]);
    std::ifstream fin(parseString(json["data"]));
    if (!fin.is_open()) {
        std::cerr << "Failed to open material file: " << json["data"].asString() << std::endl;
        exit(1);
    }

    Json::Value data;
    fin >> data;

    density = parseFloat(data["density"]);
    Vector4f stretchingData[2][5];
    for (int i = 0; i < 4; i++)
        stretchingData[0][0](i) = parseFloat(data["stretching"][0][i]);
    for (int i = 1; i < 5; i++)
        stretchingData[0][i] = stretchingData[0][0];
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 4; j++)
            stretchingData[1][i](j) = parseFloat(data["stretching"][i + 1][j]);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++) {
                Matrix2x2f G;
                G(0, 0) = static_cast<float>(i) / N - 0.25f;
                G(1, 1) = static_cast<float>(j) / N - 0.25f;
                G(0, 1) = G(1, 0) = static_cast<float>(k) / N;
                stretchingSamples[i][j][k] = calculateStretchingSample(G, stretchingData) * thicken;
            }

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 5; j++)
            bendingSamples[i][j] = parseFloat(data["bending"][i][j]) * thicken;

    fin.close();
}

Material::~Material() {}

Vector4f Material::calculateStretchingSample(const Matrix2x2f& G, const Vector4f (&data)[2][5]) const {
    Eigen::SelfAdjointEigenSolver<Matrix2x2f> evd;
    evd.compute(2.0f * G);
    Vector2f eigenValues;
    Matrix2x2f eigenVectors;
    for (int i = 0; i < 2; i++) {
        eigenValues(i) = evd.eigenvalues()(1 - i);
        eigenVectors.col(i) = evd.eigenvectors().col(1 - i);
    }
    if (eigenValues(0) == eigenValues(1) && eigenVectors(0, 0) == 0.0f && eigenVectors(1, 1) == 0.0f)
        eigenVectors = Matrix2x2f::Identity();
    if (eigenVectors(0, 0) < 0.0f)
        eigenVectors.col(0) = -eigenVectors.col(0);
    if (eigenVectors(1, 1) < 0.0f)
        eigenVectors.col(1) = -eigenVectors.col(1);

    float strainWeight = (std::sqrt(eigenValues(0) + 1.0f) - 1.0f) * 6.0f;
    strainWeight = std::clamp(strainWeight, 0.0f, 1.0f - 1e-6f);
    int strainId = static_cast<int>(strainWeight);
    strainWeight -= strainId;

    float angleWeight = std::abs(std::atan2(eigenVectors(1, 0), eigenVectors(0, 0))) / M_PI * 8.0f;
    angleWeight = std::clamp(angleWeight, 0.0f, 4.0f - 1e-6f);
    int angleId = static_cast<int>(angleWeight);
    angleWeight -= angleId;

    float weights[2][2];
    weights[0][0] = (1.0f - strainWeight) * (1.0f - angleWeight);
    weights[0][1] = (1.0f - strainWeight) * angleWeight;
    weights[1][0] = strainWeight * (1.0f - angleWeight);
    weights[1][1] = strainWeight * angleWeight;

    Vector4f ans;
    ans.setZero();
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            ans += data[strainId + i][angleId + j] * weights[i][j];
    for (int i = 0; i < 4; i++)
        ans(i) = std::max(ans(i), 0.0f);
    ans *= 2.0f;

    return ans;
}

float Material::getDensity() const {
    return density;
}

float Material::getThicken() const {
    return thicken;
}

Vector4f Material::stretchingStiffness(const Matrix2x2f& G) const {
    float x = (G(0, 0) + 0.25f) * N;
    x = std::clamp(x, 0.0f, N - 1 - 1e-6f);
    int xi = static_cast<int>(x);
    x -= xi;

    float y = (G(1, 1) + 0.25f) * N;
    y = std::clamp(y, 0.0f, N - 1 - 1e-6f);
    int yi = static_cast<int>(y);
    y -= yi;

    float z = std::abs(G(0, 1)) * N;
    z = std::clamp(z, 0.0f, N - 1 - 1e-6f);
    int zi = static_cast<int>(z);
    z -= zi;

    float weights[2][2][2];
    weights[0][0][0] = (1.0f - x) * (1.0f - y) * (1.0f - z);
    weights[0][0][1] = (1.0f - x) * (1.0f - y) * z;
    weights[0][1][0] = (1.0f - x) * y * (1.0f - z);
    weights[0][1][1] = (1.0f - x) * y * z;
    weights[1][0][0] = x * (1.0f - y) * (1.0f - z);
    weights[1][0][1] = x * (1.0f - y) * z;
    weights[1][1][0] = x * y * (1.0f - z);
    weights[1][1][1] = x * y * z;

    Vector4f ans;
    ans.setZero();
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                ans += stretchingSamples[xi + i][yi + j][zi + k] * weights[i][j][k];

    return ans;
}

float Material::bendingStiffness(float length, float angle, float area, const Vector3f& d) const {
    float curv = length * angle / area;
    float alpha = 0.5f * curv;

    float biasWeight = std::abs(std::atan2(d(1), d(0))) * 4.0f / M_PI;
    if (biasWeight > 2.0f)
        biasWeight = 4.0f - biasWeight;
    biasWeight = std::clamp(biasWeight, 0.0f, 2.0f - 1e-6f);
    int biasId = static_cast<int>(biasWeight);
    biasWeight -= biasId;

    float valueWeight = 0.2f * alpha;
    valueWeight = std::min(valueWeight, 4.0f - 1e-6f);
    int valueId = static_cast<int>(valueWeight);
    valueId = std::max(valueId, 0);
    valueWeight -= valueId;

    float weights[2][2];
    weights[0][0] = (1.0f - biasWeight) * (1.0f - valueWeight);
    weights[0][1] = (1.0f - biasWeight) * valueWeight;
    weights[1][0] = biasWeight * (1.0f - valueWeight);
    weights[1][1] = biasWeight * valueWeight;

    float ans;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            ans += bendingSamples[biasId + i][valueId + j] * weights[i][j];
    ans = std::max(ans, 0.0f);

    return ans;
}