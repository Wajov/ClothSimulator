#include "Renderer.hpp"

Renderer::Renderer(int width, int height, const std::string& path) :
    width(width),
    height(height),
    press(false),
    lastX(INT_MAX),
    lastY(INT_MAX),
    scaling(1.0f),
    rotation(Matrix4x4f::Identity()) {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    window = glfwCreateWindow(width, height, "ClothSimulator", nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(window);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int width, int height) {
        static_cast<Renderer*>(glfwGetWindowUserPointer(window))->framebufferSizeCallback(window, width, height);
    });
    glfwSetMouseButtonCallback(window, [](GLFWwindow* window, int button, int action, int mods) {
        static_cast<Renderer*>(glfwGetWindowUserPointer(window))->mouseButtonCallback(window, button, action, mods);
    });
    glfwSetCursorPosCallback(window, [](GLFWwindow* window, double x, double y) {
        static_cast<Renderer*>(glfwGetWindowUserPointer(window))->cursorPosCallback(window, x, y);
    });
    glfwSetScrollCallback(window, [](GLFWwindow* window, double x, double y) {
        static_cast<Renderer*>(glfwGetWindowUserPointer(window))->scrollCallback(window, x, y);
    });
    glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
        static_cast<Renderer*>(glfwGetWindowUserPointer(window))->keyCallback(window, key, scancode, action, mods);
    });

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return;
    }
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.0f, 1.0f);

    edgeShader = new Shader("shader/VertexShader.glsl", "shader/EdgeFragmentShader.glsl");
    faceShader = new Shader("shader/VertexShader.glsl", "shader/FaceFragmentShader.glsl");
    simulator = new Simulator(path);
}

Renderer::~Renderer() {
    glfwTerminate();
}

void Renderer::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void Renderer::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
        press = true;
    else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
        press = false;
}

void Renderer::cursorPosCallback(GLFWwindow* window, double x, double y) {
    if (press && lastX != INT_MIN && lastY != INT_MIN) {
        Vector3f a = (Vector3f(static_cast<float>(lastX) / width - 0.5f, 0.5f - static_cast<float>(lastY) / height, 1.0f)).normalized();
        Vector3f b = (Vector3f(static_cast<float>(x) / width - 0.5f, 0.5f - static_cast<float>(y) / height, 1.0f)).normalized();
        Vector3f axis = a.cross(b);
        float angle = a.dot(b);
        rotation = rotate(axis, 10.0f * std::acos(angle)) * rotation;
    }

    lastX = (int)x;
    lastY = (int)y;
}

void Renderer::scrollCallback(GLFWwindow* window, double x, double y) {
    scaling += 0.1f * (float)y;
    scaling = std::max(scaling, 0.01f);
}

void Renderer::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void Renderer::render() const {
    while (!glfwWindowShouldClose(window)) {
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        simulator->update();

        float lightPower = 30.0f;
        Vector3f lightPosition(3.0f, 3.0f, 3.0f), cameraPosition(0.0f, 0.0f, 3.0f);
        Matrix4x4f model, view, projection;
        model = scale(scaling) * rotation;
        view = lookAt(cameraPosition, Vector3f(0.0f, 0.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f));
        projection = perspective(45.0f, static_cast<float>(width) / height, 0.1f, 100.0f);

        edgeShader->use();
        edgeShader->setMat4("model", model);
        edgeShader->setMat4("view", view);
        edgeShader->setMat4("projection", projection);
        simulator->renderEdge();

        faceShader->use();
        faceShader->setMat4("model", model);
        faceShader->setMat4("view", view);
        faceShader->setMat4("projection", projection);
        faceShader->setFloat("lightPower", lightPower);
        faceShader->setVec3("lightPosition", lightPosition);
        faceShader->setVec3("cameraPosition", cameraPosition);
        simulator->renderFace();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}
