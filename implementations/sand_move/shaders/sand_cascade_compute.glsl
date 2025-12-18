#version 430

layout(local_size_x = 32, local_size_y = 32) in;

// --- BINDINGS ---
layout(std430, binding = 0) buffer BedrockSlabs { uint bedrock_slabs[]; };
layout(std430, binding = 1) buffer SandSlabs { uint sand_slabs[]; };
layout(std430, binding = 7) buffer Obstacles { uint obstacles[]; }; 

// --- UNIFORMS ---
uniform int N;
uniform float cell_size_m;
uniform float tan_repose_angle; 
uniform float transfer_rate;    

// --- CONSTANTES ---
const float SQRT2 = 1.41421356;

// --- FUNCIONES AUXILIARES ---
int idx(int x, int y) {
    return y * N + x;
}

float getHeight(int x, int y) {
    int i = idx(x, y);
    return float(bedrock_slabs[i]) + float(sand_slabs[i]);
}

void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= N || p.y >= N) return;

    int idx_p = idx(p.x, p.y);
    
    // 1. SI SOY UN OBSTACULO, NO HAGO NADA
    if (obstacles[idx_p] > 0) return;

    // Si no hay arena, no puede haber avalancha
    uint my_sand = sand_slabs[idx_p];
    if (my_sand == 0) return;

    float H_p = getHeight(p.x, p.y);

    float max_slope_diff = 0.0;
    ivec2 target_neighbor = ivec2(-1, -1);
    float target_dist = 1.0;

    // Recorremos los 8 vecinos
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue; 

            ivec2 n = p + ivec2(dx, dy);

            // Chequeo de limites del mapa
            if (n.x < 0 || n.x >= N || n.y < 0 || n.y >= N) continue;

            // 2. SI EL VECINO ES UN OBSTACULO, NO PUEDO CAER AHI
            int idx_n = idx(n.x, n.y);
            if (obstacles[idx_n] > 0) continue;

            float H_n = getHeight(n.x, n.y);
            float diff = H_p - H_n;

            // Solo nos interesa si el vecino esta mas abajo
            if (diff > 0.0) {
                float dist_factor = (dx != 0 && dy != 0) ? SQRT2 : 1.0;
                float dist_meters = dist_factor * cell_size_m;

                float slope = diff / dist_meters;

                if (slope > tan_repose_angle && slope > max_slope_diff) {
                    max_slope_diff = slope;
                    target_neighbor = n;
                    target_dist = dist_meters;
                }
            }
        }
    }

    // Si encontramos un vecino valido
    if (target_neighbor.x != -1) {
        float diff_allowed = tan_repose_angle * target_dist;
        float current_diff = H_p - getHeight(target_neighbor.x, target_neighbor.y);
        float excess_height = current_diff - diff_allowed;

        if (excess_height > 0.0) {
            float move_amount_f = (excess_height * 0.5) * transfer_rate;
            uint move_amount = uint(max(1.0, move_amount_f));
            move_amount = min(move_amount, my_sand);

            if (move_amount > 0) {
                atomicAdd(sand_slabs[idx_p], uint(-int(move_amount)));
                atomicAdd(sand_slabs[idx(target_neighbor.x, target_neighbor.y)], uint(move_amount));
            }
        }
    }
}