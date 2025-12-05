#version 430

layout(local_size_x = 32, local_size_y = 32) in;  // 32*32 hilos por work group

layout(std430, binding = 0) buffer bedrock 
{
    uint bedrock_slabs[];
};

layout(std430, binding = 1) buffer sand 
{
    uint sand_slabs[];
};

layout(std430, binding = 2) buffer WindHeightField 
{
    vec2 wind_height_field[];
};

layout(std430, binding = 3) buffer WindField 
{
    vec2 wind_field[];
};

uniform int N;
uniform float k_W = 0.005;
uniform float k_H_50 = 5.0;
uniform float k_H_200 = 30.0;

// V(p) = A(p) * (1 + k_W*H(p))
// W(p) = 0.2 * (F_50(p) o V(p)) + 0.8 * (F_200(p) o V(p))
// F_i(p) o V(p) = (1 - a) * V(p) + a * k_H_i * grad(H_i_perp(p))
// a = grad(H_i(p))

// total height
float H(int x, int y)
{
    return float(sand_slabs[x*N + y]) + float(bedrock_slabs[x*N + y]);
}

// wind height field
vec2 A(int x, int y)
{
    return wind_height_field[x*N + y].xy;
}

// venturi effect
vec2 V(int x, int y)
{
    return A(x, y) * (1 + k_W * H(x, y));
}

vec2 gradH(int x, int y) 
{
    // manejo de bordes: clamp
    int xm1 = max(x-1, 0);
    int xp1 = min(x+1, N-1);
    int ym1 = max(y-1, 0);
    int yp1 = min(y+1, N-1);

    float dx = (H(xp1, y) - H(xm1, y)) * 0.5;
    float dy = (H(x, yp1) - H(x, ym1)) * 0.5;

    return vec2(dx, dy);
}

vec2 gradH_perp(int x, int y) 
{
    vec2 g = gradH(x, y);
    return vec2(-g.y, g.x);
}

float a(int x, int y) {
    vec2 g = gradH(x, y);
    return length(g);
}

vec2 F_50oV(int x, int y)
{
    float alpha = a(x, y);
    return (1.0 - alpha) * V(x, y) + alpha * k_H_50 * gradH_perp(x, y);
}

vec2 F_200oV(int x, int y)
{
    float alpha = a(x, y);
    return (1.0 - alpha) * V(x, y) + alpha * k_H_200 * gradH_perp(x, y);
}

// Final wind field
vec2 W(int x, int y)
{
    return 0.2 * F_50oV(x, y) + 0.8 * F_200oV(x, y);
}

void main() 
{
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);

    if (coords.x*coords.y > N*N) return;

    wind_field[coords.x*N + coords.y] = W(coords.x, coords.y);
}