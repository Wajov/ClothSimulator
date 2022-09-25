#ifndef RENDERER_HPP
#define RENDERER_HPP

#include <climits>
#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "TypeHelper.hpp"
#include "TransformHelper.hpp"
#include "Shader.hpp"
#include "Simulator.hpp"

class Renderer {
private:
    bool press;
    int width, height, lastX, lastY;
    float scaling;
    Matrix4x4f rotation;
    GLFWwindow* window;
    Shader* edgeShader, * faceShader;
    Simulator *simulator;
    void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    void cursorPosCallback(GLFWwindow* window, double x, double y);
    void scrollCallback(GLFWwindow* window, double x, double y);
    void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

public:
    Renderer(int width, int height, const std::string& path);
    ~Renderer();
    void render() const;
};

#endif