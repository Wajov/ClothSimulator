#version 330 core

in vec3 vertexPosition;
in vec3 vertexNormal;
in vec2 vertexUV;

uniform vec3 color;
uniform vec3 cameraPosition;
uniform vec3 lightDirection;
uniform int selectedFace;

void main() {
    vec3 ambientColor = 0.5 * color;
    vec3 diffuseColor = 0.5 * color;

    vec3 ambient = ambientColor;

    vec3 N = normalize(vertexNormal);
    vec3 L = normalize(lightDirection);
    vec3 diffuse =  diffuseColor * abs(dot(N, L));

    if (selectedFace == gl_PrimitiveID)
        gl_FragColor = vec4(1, 0, 0, 1);
    else
        gl_FragColor = vec4(ambient + diffuse, 1);
}