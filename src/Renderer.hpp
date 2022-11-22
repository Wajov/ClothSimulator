#ifndef RENDERER_HPP
#define RENDERER_HPP

#include <cmath>
#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "MathHelper.cuh"
#include "Vector.cuh"
#include "Matrix.cuh"
#include "Transform.hpp"
#include "Simulator.cuh"

class Renderer {
private:
    bool press, pause;
    int width, height, pressX, pressY;
    double lastX, lastY, scaling;
    Matrix4x4f rotation;
    GLFWwindow* window;
    Simulator* simulator;
    void framebufferSizeCallback(int width, int height);
    void mouseButtonCallback(int button, int action, int mods);
    void cursorPosCallback(double x, double y);
    void scrollCallback(double x, double y);
    void keyCallback(int key, int scancode, int action, int mods);

public:
    Renderer(int width, int height, const std::string& path);
    ~Renderer();
    void render() const;
};

#endif