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

// Sticky y erosion masks (separadas)
layout(std430, binding = 5) buffer StickyMask {
    float sticky_mask[]; // [0,1], kb incluido
};

layout(std430, binding = 6) buffer ErosionMask {
    float erosion_mask[]; // [0,1]
};

uniform int N;                // tamaño de la grilla (N x N)
uniform float cell_size_m;    // tamaño de celda en metros (p.ej. 1.0)
uniform float h_max;          // límite superior de altura de cliff (metros o “unidades” de H)
uniform float kb;             // bias (p.ej. 0.1)
uniform float slope_deg_thresh;  // umbral de 55.0 deg
uniform int R_s;       // límite de pasos al retroceder (p.ej. 10)

// Función para indexar fila-columna (y, x)
int idx(int x, int y) {
    return y * N + x;
}

// Altura total
float H(int x, int y) {
    return float(sand_slabs[idx(x,y)]) + float(bedrock_slabs[idx(x,y)]);
}

// Dirección upwind a partir de un campo de viento W; si no lo tienes, usa wind_height_field
vec2 windAt(int x, int y) {
    // Usa wind_field si ya lo calculaste; si no, usa wind_height_field como aproximación
    return wind_field[idx(x,y)];
    // return wind_height_field[idx(x,y)]; // alternativa si W todavía no existe
}

void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= N || p.y >= N) return;

    int px = p.x;
    int py = p.y;

    // Dirección upwind (contra el viento)
    vec2 Wp = windAt(px, py);
    float windLen = length(Wp);
    if (windLen < 1e-6) {
        // Sin viento: no marcamos sticky ni erosión
        sticky_mask[idx(px,py)] = kb;       // solo bias mínimo
        erosion_mask[idx(px,py)] = 0.0;
        return;
    }
    vec2 upwind = -Wp / windLen;

    // Paso 1: detectar cliff cell contra el viento (primer vecino con “dropoff” > umbral)
    // Caminamos al menos 1 celda contra el viento para comparar alturas
    int cliff_x = px;
    int cliff_y = py;
    bool found_cliff = false;

    {
        // Primer vecino “inmediato” contra el viento
        int qx = clamp(px + int(round(upwind.x)), 0, N-1);
        int qy = clamp(py + int(round(upwind.y)), 0, N-1);

        float Hp = H(px, py);
        float Hq = H(qx, qy);

        // Distancia horizontal en metros entre p y q (una celda salvo diagonales discretas)
        float horiz_dist = length(vec2(qx - px, qy - py)) * cell_size_m;
        horiz_dist = max(horiz_dist, 1e-6);

        float dh = Hp - Hq; // "dropoff" respecto a vecino contra el viento
        float slope_rad = atan(dh / horiz_dist);
        float slope_deg = degrees(slope_rad);

        if (slope_deg > slope_deg_thresh) {
            found_cliff = true;
            cliff_x = px;
            cliff_y = py;
        }
    }

    // Si la celda actual no es cliff, avanzamos más contra el viento para buscar la primera que cumpla
    if (!found_cliff) {
        int cx = px;
        int cy = py;
        float Hp = H(cx, cy);

        for (int step = 1; step <= max_steps; ++step) {
            int nx = clamp(px + int(round(upwind.x * step)), 0, N-1);
            int ny = clamp(py + int(round(upwind.y * step)), 0, N-1);

            float Hn = H(nx, ny);
            float horiz_dist = length(vec2(nx - cx, ny - cy)) * cell_size_m;
            horiz_dist = max(horiz_dist, 1e-6);

            float dh = Hp - Hn; // dropoff respecto al vecino “anterior” en la marcha
            float slope_deg = degrees(atan(dh / horiz_dist));

            if (slope_deg > slope_deg_thresh) {
                found_cliff = true;
                cliff_x = nx;
                cliff_y = ny;
                break;
            }

            cx = nx;
            cy = ny;
            Hp = H(cx, cy);
        }
    }

    // Si no hay cliff, no hay sticky/erosión especial
    if (!found_cliff) {
        sticky_mask[idx(px,py)] = kb; // solo bias mínimo
        erosion_mask[idx(px,py)] = 0.0;
        return;
    }

    // Paso 2: altura del cliff h_o como diferencial con el vecino más cercano contra el viento
    {
        // vecino inmediato contra el viento desde el cliff
        int nx = clamp(cliff_x + int(round(upwind.x)), 0, N-1);
        int ny = clamp(cliff_y + int(round(upwind.y)), 0, N-1);

        float Hc = H(cliff_x, cliff_y);
        float Hn = H(nx, ny);

        float h_o = Hc - Hn;         // diferencial de altura del cliff
        h_o = max(h_o, 0.0);         // evitar negativos
        h_o = min(h_o, h_max);       // límite superior

        // Distancias en metros donde se forman sticky y erosión
        float d_min_sticky = 0.4 * h_o;
        float d_max_sticky = 2.0 * h_o;

        // Si h_o ~ 0, no hay efecto
        if (d_max_sticky <= 1e-6) {
            sticky_mask[idx(px,py)] = kb;
            erosion_mask[idx(px,py)] = 0.0;
            return;
        }

        // Paso 3: recorrer contra el viento y asignar máscaras según distancia desde la cliff cell
        float sticky_val = kb;
        float erosion_val = 0.0;

        // Distancia desde la cliff cell a esta celda p (en metros)
        float d_from_cliff = length(vec2(px - cliff_x, py - cliff_y)) * cell_size_m;

        // Erosión si d <= 0.4 h_o
        if (d_from_cliff <= d_min_sticky) {
            erosion_val = 1.0;        // marcamos fuerte erosión
            sticky_val = kb;          // sticky solo bias
        }
        // Sticky si d ∈ [0.4 h_o, 2 h_o]
        else if (d_from_cliff <= d_max_sticky) {
            float t = (d_from_cliff - d_min_sticky) / max(d_max_sticky - d_min_sticky, 1e-6);
            // “inverse percentage through the range”: primera sticky (cerca de 0.4 h_o) -> 1+kb, más lejos -> 0+kb
            // Interpretamos que el valor final se clampa a [0,1] y sumamos kb como sesgo:
            float base = 1.0 - t;              // 1 en el inicio, 0 al final
            sticky_val = kb + base * (1.0 - kb);
            erosion_val = 0.0;
        } else {
            // fuera del rango sticky
            sticky_val = kb;
            erosion_val = 0.0;
        }

        // Guarda las máscaras
        sticky_mask[idx(px,py)] = clamp(sticky_val, 0.0, 1.0);
        erosion_mask[idx(px,py)] = clamp(erosion_val, 0.0, 1.0);
    }
}