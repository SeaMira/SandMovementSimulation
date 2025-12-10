#version 430
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 globPosition;
layout(location = 3) in uint sand_slabs;
layout(location = 4) in uint bedrock_slabs;
layout(location = 5) in uint obstacles;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform int topheight;

out vec3 fragColor;
out vec3 vNormal;
out vec3 FragPos;

void main()
{
    vec3 pos = position + vec3(globPosition.x, 0.5, globPosition.y);

    if (obstacles == uint(1) && pos.y > 0.0)
        pos.y = topheight;
    else if (pos.y > 0.0) 
        pos.y *= sand_slabs + bedrock_slabs;
    else 
        pos.y = bedrock_slabs;
    
    FragPos = vec3(model * vec4(pos, 1.0));
    if (obstacles != uint(0))
        fragColor = vec3(0.60, 0.53, 0.39);
    else if (sand_slabs != uint(0))
        fragColor = vec3(0.94, 0.87, 0.73);
    else
        fragColor = vec3(0.77, 0.70, 0.56);
    vNormal = normal;
    gl_Position = projection * view * model * vec4(pos, 1.0f);
}