#include "Motion.cuh"

Motion::Motion(const Json::Value& json) :
    index(0) {
    n = json.size();
    t.resize(n);
    x.resize(n);
    v.resize(n);
    for (int i = 0; i < n; i++) {
        t[i] = parseFloat(json[i]["time"]);
        x[i] = Transformation(json[i]["transform"]);
    }
    
    v[0] = v[n - 1] = 0.0f * Transformation();
    for (int i = 1; i < n - 1; i++)
        v[i] = (x[i + 1] - x[i - 1]) / (t[i + 1] - t[i - 1]);
    updateCoefficients();
}

Motion::~Motion() {}

void Motion::updateCoefficients() {
    float dt = t[index + 1] - t[index];
    Transformation dx = x[index + 1] - x[index];
    a = dt * (v[index + 1] + v[index])  - 2.0f * dx;
    b = 3.0f * dx - dt * (v[index + 1] + 2.0f * v[index]);
    c = dt * v[index];
    d = x[index];
}

Transformation Motion::computeTransformation(float time) {
    bool flag = false;
    while (index < n - 1 && t[index + 1] < time) {
        index++;
        flag = true;
    }
    if (index == n - 1)
        return x[n - 1];
    
    if (flag)
        updateCoefficients();

    float s = (time - t[index]) / (t[index + 1] - t[index]);
    float s2 = s * s;
    float s3 = s * s2;
    return a * s3 + b * s2 + c * s + d;
}