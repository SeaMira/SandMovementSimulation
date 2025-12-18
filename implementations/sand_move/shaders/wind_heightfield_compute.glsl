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

uniform int N;

// total height
float H(int x, int y)
{
    return float(sand_slabs[x*N + y]) + float(bedrock_slabs[x*N + y]);
}

void main() 
{
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);

    if (coords.x*coords.y > N*N) return;

    // celda base
    int px = coords.x;
    int py = coords.y;
    ///////////////////
    float Hp = max(H(px, py), 1.0);

    float alpha = atan(float(py), float(px) + 0.01);
    wind_height_field[px*N + py] = log(Hp)*vec2(cos(alpha), sin(alpha));
    ///////////////////
    
    // vec2 wind_dir = normalize(vec2(1.0, 0.0)); 
    
    // Magnitud base logarítmica de la altura (como tenías)
    // float Hp = max(H(px, py), 1.0);
    
    // wind_height_field[px*N + py] = wind_dir;

}