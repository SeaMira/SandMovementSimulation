from noise import pnoise2
import numpy as np
import random

def generar_alturas(N, scale=0.1, octaves=3, persistence=0.5, lacunarity=2.0, base=0, top_height=5, tolerance=-0.5):
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
            alturas[i, j] = (h + 1) * 0.5 * top_height if h > tolerance else 0
    return alturas


def generar_obstaculos(N, scale=0.03, octaves=2, persistence=0.5, lacunarity=2.0, seed=None, threshold=0.2):
    """
    Genera un mapa de obstáculos agrupados (blobs) usando Perlin Noise.
    Returns: Array plano 1D de uint32 con 0s y 1s.
    """
    if seed is None:
        seed = random.randint(0, 100)
        
    # Usamos un array plano directamente porque el SSBO lo prefiere así
    obstaculos = np.zeros(N * N, dtype=np.uint32)
    
    for i in range(N):
        for j in range(N):
            # Generamos ruido [-1, 1]
            # Usamos 'seed' como base para que los obstáculos no coincidan
            # exactamente con las dunas de arena (si no quieres).
            h = pnoise2(i * scale, 
                        j * scale, 
                        octaves=octaves, 
                        persistence=persistence, 
                        lacunarity=lacunarity, 
                        base=seed)
            
            # APLICAMOS EL UMBRAL (THRESHOLD)
            # Si el valor del ruido es mayor que el umbral, ponemos un 1 (obstáculo)
            # Si scale es bajo (ej. 0.02), los grupos serán grandes.
            if h > threshold:
                obstaculos[i * N + j] = 1
            else:
                obstaculos[i * N + j] = 0
                
    return obstaculos