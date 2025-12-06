#version 430

layout(local_size_x = 32, local_size_y = 32) in;

// Terreno base
layout(std430, binding = 0) buffer bedrock {
    uint bedrock_slabs[];
};

layout(std430, binding = 1) buffer sand {
    uint sand_slabs[];
};

// Campo de viento “base” (A) y campo de viento proyectado (W) si lo necesitas
layout(std430, binding = 2) buffer WindHeightField {
    vec2 wind_height_field[]; // A(p) precargado o generado
};

layout(std430, binding = 3) buffer WindField {
    vec2 wind_field[]; // W(p), si lo calculas en otra pasada
};

layout(std430, binding = 4) buffer WindShadowing 
{
    float wind_shadowing[];
};

// Sticky y erosion masks (separadas)
layout(std430, binding = 5) buffer StickyMask {
    float sticky_mask[]; // [0,1], kb incluido
};

layout(std430, binding = 6) buffer ErosionMask {
    float erosion_mask[]; // [0,1]
};

layout(std430, binding = 7) buffer Obstacles 
{
    uint obstacles[];
};

uniform uint sand_transport_block_count;
uniform int N;                // tamaño de la grilla (N x N)
uniform int R_s;       // límite de pasos al retroceder (p.ej. 10)
uniform float cell_size_m;    // tamaño de celda en metros (p.ej. 1.0)


struct Cell
{
    uint bedrock;
    uint sand;
    uint obstacle;
    float sticky;
    float erosion;
    float wind_shadow;
    vec2 wind;
};

Cell makeCell(int idx) 
{
    Cell c;
    c.bedrock = bedrock_slabs[idx];
    c.sand = sand_slabs[idx];
    c.obstacle = obstacles[idx];
    c.sticky = sticky_mask[idx];
    c.erosion = erosion_mask[idx];
    c.wind = wind_field[idx];
    c.wind_shadow = wind_shadowing[idx];
    return c;
}

float rand(vec2 co) 
{
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

// Función para indexar fila-columna (y, x)
int idx(int x, int y) {
    return y * N + x;
}

// Altura total
float H(int x, int y) {
    return float(sand_slabs[idx(x,y)]) + float(bedrock_slabs[idx(x,y)]);
}

void main()
{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= N || p.y >= N) return;

    const int px = p.x;
    const int py = p.y;
    const int id_pos = idx(px, py);
    Cell C_p = makeCell(id_pos);

    if (C_p.obstacle > uint(0) || rand(vec2(p)) < C_p.sticky || rand(vec2(p)) < C_p.wind_shadow)
        return;

    uint transported_sand = sand_transport_block_count;
    if (C_p.erosion > 0.0)
        transported_sand = uint(float(transported_sand)*(1.0 + C_p.erosion));

    vec2 stepVec = normalize(C_p.wind);
    bool deposited = false;
    C_p.sand -= transported_sand;
    sand_slabs[id_pos] -= transported_sand;
    ivec2 q = p;
    int count = 0;

    while (!deposited || count < R_s)
    {
        q = ivec2(clamp(q.x + int(round(stepVec.x)), 0, N-1), clamp(q.x + int(round(stepVec.y)), 0, N-1));
        Cell C_q = makeCell(idx(q.x, q.y));
        if (C_q.obstacle > 0)
        {
            q = ivec2(clamp(q.x - int(round(stepVec.x)), 0, N-1), clamp(q.x - int(round(stepVec.y)), 0, N-1));
            C_q = makeCell(idx(q.x, q.y));
            C_q.sand += transported_sand;
            sand_slabs[idx(q.x, q.y)] += transported_sand;
            return;
        }
        if (rand(vec2(q)) < C_q.sticky || rand(vec2(q)) < C_q.wind_shadow)
        {
            deposited = true;
            C_q.sand += transported_sand;
            sand_slabs[idx(q.x, q.y)] += transported_sand;
            return;
        }
        count++;
    }

    C_p.sand += transported_sand;
    sand_slabs[id_pos] += transported_sand;
}