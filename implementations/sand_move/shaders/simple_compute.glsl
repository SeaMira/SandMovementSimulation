#version 430

layout(local_size_x = 64) in;  // 64 hilos por work group

struct CubeData {
    vec2 position;
};

layout(std430, binding = 0) buffer Cubes {
    CubeData cubes[];
};

// uniform float time;

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= cubes.length()) return;

    // Movimiento simple: altura "flotante"
    // cubes[id].position.y += sin(time + float(id)) * 0.01f;
}
