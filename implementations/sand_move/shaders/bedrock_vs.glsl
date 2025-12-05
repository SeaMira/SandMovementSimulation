#version 330
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 globPosition;
layout(location = 4) in uint bedrock_slabs;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 fragColor;
out vec3 vNormal;
out vec3 FragPos;

void main()
{
    vec3 pos = position + vec3(globPosition.x, 0.5, globPosition.y);

    if (pos.y > 0.0) {
        pos.y *= bedrock_slabs;
    }
    FragPos = vec3(model * vec4(pos, 1.0));
    fragColor = vec3(0.77, 0.70, 0.56);
    vNormal = normal;
    gl_Position = projection * view * model * vec4(pos, 1.0f);
}