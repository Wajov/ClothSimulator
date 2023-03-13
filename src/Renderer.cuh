#ifndef RENDERER_CUH
#define RENDERER_CUH

#include <cmath>
#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "MathHelper.cuh"
#include "Vector.cuh"
#include "Matrix.cuh"
#include "Transform.cuh"

class Renderer {
private:
    bool press, pause;
    int width, height, pressX, pressY;
    double lastX, lastY, scaling;
    Vector3f lightDirection, cameraPosition;
    Matrix4x4f rotation;
    GLFWwindow* window;
    void framebufferSizeCallback(int width, int height);
    void mouseButtonCallback(int button, int action, int mods);
    void cursorPosCallback(double x, double y);
    void scrollCallback(double x, double y);
    void keyCallback(int key, int scancode, int action, int mods);

public:
    Renderer(int width, int height);
    ~Renderer();
    GLFWwindow* getWindow() const;
    bool getPause() const;
    Vector3f getLightDirection() const;
    Vector3f getCameraPosition() const;
    Matrix4x4f getModel() const;
    Matrix4x4f getView() const;
    Matrix4x4f getProjection() const;
};

#endif