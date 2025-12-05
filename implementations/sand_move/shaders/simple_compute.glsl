#version 430

layout(local_size_x = 32, local_size_y = 32) in;  // 64 hilos por work group

layout(std430, binding = 0) buffer sand 
{
    uint sand_slabs[];
};

uniform int N;

void main() 
{
    ivec2 coords = ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);

    if (coords.x*coords.y > N*N) return;

    // sand_slabs[N*coords.x + coords.y] += uint(1);


    // Movimiento simple: altura "flotante"
    // cubes[id].position.y += sin(time + float(id)) * 0.01f;
}
