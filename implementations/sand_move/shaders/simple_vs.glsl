#version 330
layout(location = 0) in vec3 position;
layout(location = 1) in vec4 globPosition;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 fragColor;

void main()
{
    vec3 pos = position + vec3(globPosition);
    fragColor = vec3(1.0f, 0.0f, 0.0f);
    gl_Position = projection * view * model * vec4(pos, 1.0f);
}