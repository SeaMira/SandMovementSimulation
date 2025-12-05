from noise import pnoise2
import numpy as np

def generar_alturas(N, scale=0.1, octaves=3, persistence=0.5, lacunarity=2.0, base=0, top_height=5):
    alturas = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            # perlin devuelve valores [-1, 1], los reescalamos
            h = pnoise2(i * scale,
                        j * scale,
                        octaves=octaves,
                        persistence=persistence,
                        lacunarity=lacunarity,
                        base=base)
            # normalizamos a [0,1] y escalamos a [0, top_height]
            alturas[i, j] = (h + 1) * 0.5 * top_height if h > -0.5 else 0
    return alturas