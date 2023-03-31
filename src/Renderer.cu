#include "Renderer.cuh"

Renderer::Renderer(int width, int height) :
    width(width),
    height(height),
    leftPress(false),
    rightPress(false),
    pause(true),
    leftX(INFINITY),
    leftY(INFINITY),
    rightX(INFINITY),
    rightY(INFINITY),
    scaling(1.0f),
    rotation(1.0f),
    lightDirection(0.0f, 0.0f, 1.0f),
    cameraPosition(0.0f, -0.25f, 2.0f) {
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
}

Renderer::~Renderer() {
    glfwTerminate();
}

Matrix4x4f Renderer::scale(float scaling) const {
    Matrix4x4f ans;

    ans(0, 0) = ans(1, 1) = ans(2, 2) = scaling;
    ans(3, 3) = 1.0f;

    return ans;
}

Matrix4x4f Renderer::rotate(const Vector3f& v, float angle) const {
    Vector3f axis = v.normalized();
    float s = sin(angle);
    float c = cos(angle);
    Matrix4x4f ans;

    ans(0, 0) = (1.0f - c) * axis(0) * axis(0) + c;
    ans(0, 1) = (1.0f - c) * axis(1) * axis(0) - s * axis(2);
    ans(0, 2) = (1.0f - c) * axis(2) * axis(0) + s * axis(1);

    ans(1, 0) = (1.0f - c) * axis(0) * axis(1) + s * axis(2);
    ans(1, 1) = (1.0f - c) * axis(1) * axis(1) + c;
    ans(1, 2) = (1.0f - c) * axis(2) * axis(1) - s * axis(0);

    ans(2, 0) = (1.0f - c) * axis(0) * axis(2) - s * axis(1);
    ans(2, 1) = (1.0f - c) * axis(1) * axis(2) + s * axis(0);
    ans(2, 2) = (1.0f - c) * axis(2) * axis(2) + c;

    ans(3, 3) = 1.0f;

    return ans;
}

Matrix4x4f Renderer::translate(const Vector3f& v) const {
    Matrix4x4f ans(1.0f);

    ans(0, 3) = v(0);
    ans(1, 3) = v(1);
    ans(2, 3) = v(2);

    return ans;
}

Matrix4x4f Renderer::lookAt(const Vector3f& position, const Vector3f& center, const Vector3f& up) const {
    Vector3f f = (center - position).normalized();
    Vector3f s = f.cross(up).normalized();
    Vector3f u = s.cross(f);
    Matrix4x4f ans;

    ans(0, 0) = s(0);
    ans(0, 1) = s(1);
    ans(0, 2) = s(2);
    ans(0, 3) = -s.dot(position);

    ans(1, 0) = u(0);
    ans(1, 1) = u(1);
    ans(1, 2) = u(2);
    ans(1, 3) = -u.dot(position);

    ans(2, 0) = -f(0);
    ans(2, 1) = -f(1);
    ans(2, 2) = -f(2);
    ans(2, 3) = f.dot(position);

    ans(3, 3) = 1.0f;

    return ans;
}

Matrix4x4f Renderer::perspective(float fovy, float aspect, float zNear, float zFar) const {
    float t = tan(fovy * 0.5f);
    Matrix4x4f ans;

    ans(0, 0) = 1.0f / (aspect * t);
    ans(1, 1) = 1.0f / t;
    ans(2, 2) = -(zNear + zFar) / (zFar - zNear);
    ans(2, 3) = -2.0f * zNear * zFar / (zFar - zNear);
    ans(3, 2) = -1.0f;

    return ans;
}

void Renderer::framebufferSizeCallback(int width, int height) {
    this->width = width;
    this->height = height;
    glViewport(0, 0, width, height);
}

void Renderer::mouseButtonCallback(int button, int action, int mods) {
    double x, y;
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        leftPress = true;
        leftX = leftY = INFINITY;
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
        rightPress = true;
        rightX = rightY = INFINITY;
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE)
        rightPress = false;
    else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
        leftPress = false;
}

void Renderer::cursorPosCallback(double x, double y) {
    if (leftPress && leftX != INFINITY && leftY != INFINITY) {
        Vector3f a = (Vector3f(static_cast<float>(leftX) / width - 0.5f, 0.5f - static_cast<float>(leftY) / height, 1.0f)).normalized();
        Vector3f b = (Vector3f(static_cast<float>(x) / width - 0.5f, 0.5f - static_cast<float>(y) / height, 1.0f)).normalized();
        Vector3f axis = a.cross(b);
        float angle = a.dot(b);
        rotation = rotate(axis, 10.0f * acos(angle)) * rotation;
    }
    if (rightPress && rightX != INFINITY && rightY != INFINITY) {
        Vector3f v(static_cast<float>(x) - rightX, rightY - static_cast<float>(y), 0.0f);
        translation += 0.005f * v;
    }

    leftX = rightX = x;
    leftY = rightY = y;
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

GLFWwindow* Renderer::getWindow() const {
    return window;
}

bool Renderer::getPause() const {
    return pause;
}

Vector3f Renderer::getLightDirection() const {
    return lightDirection;
}

Vector3f Renderer::getCameraPosition() const {
    return cameraPosition;
}

Matrix4x4f Renderer::getModel() const {
    return translate(translation) * scale(static_cast<float>(scaling)) * rotation;
}

Matrix4x4f Renderer::getView() const {
    return lookAt(cameraPosition, Vector3f(0.0f, -0.25f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f));
}

Matrix4x4f Renderer::getProjection() const {
    return perspective(45.0f, static_cast<float>(width) / height, 0.1f, 100.0f);
}