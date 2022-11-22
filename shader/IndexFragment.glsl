#version 330 core

in vec3 vertexPosition;
in vec3 vertexNormal;
in vec2 vertexUV;

uniform int clothIndex;

out ivec2 index;

void main() {
    index = ivec2(clothIndex, gl_PrimitiveID);
}