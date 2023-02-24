#include "Renderer.cuh"

Renderer::Renderer(int width, int height, const std::string& path) :
    width(width),
    height(height),
    press(false),
    pause(true),
    lastX(INFINITY),
    lastY(INFINITY),
    scaling(1.0f),
    rotation(1.0f) {
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
        static_cast<Renderer*>(glfwGetWindowUserPointer(window))->framebufferSizeCallback(width, height);
    });
    glfwSetMouseButtonCallback(window, [](GLFWwindow* window, int button, int action, int mods) {
        static_cast<Renderer*>(glfwGetWindowUserPointer(window))->mouseButtonCallback(button, action, mods);
    });
    glfwSetCursorPosCallback(window, [](GLFWwindow* window, double x, double y) {
        static_cast<Renderer*>(glfwGetWindowUserPointer(window))->cursorPosCallback(x, y);
    });
    glfwSetScrollCallback(window, [](GLFWwindow* window, double x, double y) {
        static_cast<Renderer*>(glfwGetWindowUserPointer(window))->scrollCallback(x, y);
    });
    glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
        static_cast<Renderer*>(glfwGetWindowUserPointer(window))->keyCallback(key, scancode, action, mods);
    });

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return;
    }
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.0f, 1.0f);

    simulator = new Simulator(path);
}

Renderer::~Renderer() {
    glfwTerminate();
    delete simulator;
}

void Renderer::framebufferSizeCallback(int width, int height) {
    this->width = width;
    this->height = height;
    glViewport(0, 0, width, height);
}

void Renderer::mouseButtonCallback(int button, int action, int mods) {
    double x, y;
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        press = true;
        glfwGetCursorPos(window, &x, &y);
        pressX = (int)x;
        pressY = (int)y;
    } else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
        press = false;
        glfwGetCursorPos(window, &x, &y);
        if (pressX == (int)x && pressY == (int)y)
            simulator->printDebugInfo(pressX, height - pressY);
    }
}

void Renderer::cursorPosCallback(double x, double y) {
    if (press && lastX != INFINITY && lastY != INFINITY) {
        Vector3f a = (Vector3f(static_cast<float>(lastX) / width - 0.5f, 0.5f - static_cast<float>(lastY) / height, 1.0f)).normalized();
        Vector3f b = (Vector3f(static_cast<float>(x) / width - 0.5f, 0.5f - static_cast<float>(y) / height, 1.0f)).normalized();
        Vector3f axis = a.cross(b);
        float angle = a.dot(b);
        rotation = Transform::rotate(axis, 10.0f * acos(angle)) * rotation;
    }

    lastX = x;
    lastY = y;
}

void Renderer::scrollCallback(double x, double y) {
    scaling += 0.1 * y;
    scaling = max(scaling, 0.01);
}

void Renderer::keyCallback(int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    else if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        pause = !pause;
}

void Renderer::render() const {
    while (!glfwWindowShouldClose(window)) {
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        Vector3f lightDirection(0.0f, 0.0f, 1.0f), cameraPosition(0.0f, -0.25f, 2.0f);
        Matrix4x4f model, view, projection;
        model = Transform::scale(static_cast<float>(scaling)) * rotation;
        view = Transform::lookAt(cameraPosition, Vector3f(0.0f, -0.25f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f));
        projection = Transform::perspective(45.0f, static_cast<float>(width) / height, 0.1f, 100.0f);

        simulator->render(width, height, model, view, projection, cameraPosition, lightDirection);
        if (!pause)
            simulator->step();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}
