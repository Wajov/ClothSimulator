#ifndef SHADER_HPP
#define SHADER_HPP

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#include <glad/glad.h>

#include "Vector.hpp"
#include "Matrix.hpp"

class Shader {
private:
    unsigned int program;

public:
    Shader(const std::string& vertexShaderPath, const std::string& fragmentShaderPath);
    ~Shader();
    void use() const;
    void setFloat(const std::string& name, float value) const;
    void setVec3(const std::string& name, const Vector3f& value) const;
    void setMat4(const std::string& name, const Matrix4x4f& value) const;
};

#endif