import pyglet
from pyglet.window import key, mouse
import implementations.sand_move.camera as cam
from OpenGL import GL
import numpy as np
import os
from pathlib import Path
from utils.load_pipeline import load_pipeline, compute_program_pipeline
from utils.elementos import rectangulo, cubo_unitario
from utils.gl_utils import SSBO, RenderingInstance, setInstanceArrayAttribute, setInstanceArrayIAttribute
from implementations.sand_move.height_map_noise import generar_alturas, generar_obstaculos
import click
import ctypes
import time

import imgui
from imgui.integrations.pyglet import create_renderer

# --- CONFIGURACIÓN GLOBAL ---
w_width = 1920
w_height = 1080
group_size_x = 32
group_size_y = 32
N = 512 

# Parámetros iniciales
top_sand_height = 12
top_bedrock_height = 12
max_steps = 10 
cell_size_m = 1.0
h_max = 24
kb = 0.1
slope_deg_thresh = 55.0
sand_transport_block_count = 2
repose_angle = 33.0
transfer_rate = 0.25
cascade_iterations = 10

# Tiempo entre ejecuciones de lógica (en segundos)
accumulated_time = 0.0
time_to_compute_executions = 1.0

# Luces
lightDir = np.array([1.0, -1.0, 0.0], dtype=np.float32) * (1.0/(2.0**(1.0/2.0)))
lightColor = np.array([1.0, 1.0, 1.0], dtype=np.float32)

# --- FUNCIÓN PARA GENERAR DATOS (Necesaria para el reinicio) ---
def generar_datos_iniciales():
    print("Generando terreno...")
    # 1. Generar mapas de altura
    sand_heights = generar_alturas(N, scale=0.05, octaves=3, persistence=0.3, lacunarity=1.0, base=0, top_height=top_sand_height, tolerance=-0.5)
    bedrock_heights = generar_alturas(N, scale=0.15, octaves=3, persistence=0.7, lacunarity=1.0, base=1, top_height=top_bedrock_height, tolerance=-0.5)
    obstacles_data = generar_obstaculos(N, scale=0.02, threshold=0.15, seed=42) # Seed aleatoria al reiniciar

    model_matrices = []
    sand_slabs = []
    bedrock_slabs = []

    # 2. Aplanar datos para los buffers
    for i in range(N):
        for j in range(N):
            x = i - N/2
            z = j - N/2
            n_sand_slab = int(sand_heights[i, j])
            n_bedrock_slab = int(bedrock_heights[i, j])

            model_matrices.append([x, z])
            sand_slabs.append(n_sand_slab)
            bedrock_slabs.append(n_bedrock_slab)

    # 3. Convertir a Numpy Arrays con tipos correctos
    model_matrices = np.array(model_matrices, dtype=np.float32)
    sand_slabs = np.array(sand_slabs, dtype=np.uint32)
    bedrock_slabs = np.array(bedrock_slabs, dtype=np.uint32)
    obstacles_data = np.array(obstacles_data, dtype=np.uint32) # Asegurar tipo
    print("Terreno generado.")
    return model_matrices, sand_slabs, bedrock_slabs, obstacles_data

# Generamos datos iniciales globales (para setup de ventana)
model_matrices_init, sand_slabs_init, bedrock_slabs_init, obstacles_data_init = generar_datos_iniciales()
n_instances = len(model_matrices_init)



camera = cam.Camera(w_width, w_height)
seconds = 0

def setCameraUniforms(c_camera, pipeline):
    pipeline["camPos"] = c_camera.get_pos()
    pipeline["projection"] = c_camera.get_perspective().reshape(16, 1, order="F")
    pipeline["view"] = c_camera.get_view().reshape(16, 1, order="F")
    pipeline["model"] = c_camera.get_model().reshape(16, 1, order="F")

@click.command("sand_move", short_help="Ejecucion de simulación de movimiento de arena")
def sand_move():
    global seconds, transfer_rate, repose_angle, max_steps, sand_transport_block_count, kb

    # --- SETUP PYGLET ---
    config = pyglet.gl.Config(
        major_version=4, 
        minor_version=5, 
        depth_size=24, 
        double_buffer=True
    )

    window = pyglet.window.Window(width=w_width, height=w_height, caption="Sand Dunes Simulations", resizable=True, config=config)

    # Maximizar ventana si se desea (opcional, mejor que fullscreen para debug)
    # window.maximize() 

    keys = key.KeyStateHandler()
    window.push_handlers(keys)

    # --- SETUP IMGUI ---
    imgui.create_context()
    impl = create_renderer(window)

    # === SOLUCIÓN 1: ESCALAR LA UI ===
    io = imgui.get_io()
    io.font_global_scale = 1.5  # Aumenta esto (ej. 2.0) si sigue siendo pequeño

    print("Iniciando simulación de movimiento de arena...")
    # --- SHADERS ---
    # (Tu código de carga de shaders se mantiene igual)
    shader_path = Path(os.path.dirname(__file__)) / "shaders"
    print(f"Cargando shaders desde: {shader_path}")

    cube_data = cubo_unitario()
    
    sand_slabs_pipeline = load_pipeline(shader_path / "sand_vs.glsl", shader_path / "sand_fs.glsl") 
    bedrock_pipeline = load_pipeline(shader_path / "bedrock_vs.glsl", shader_path / "bedrock_fs.glsl") 

    wind_heightfield_compute = compute_program_pipeline(shader_path/"wind_heightfield_compute.glsl")
    wind_update_compute = compute_program_pipeline(shader_path/"wind_update_compute.glsl")
    sticky_mask_compute = compute_program_pipeline(shader_path/"sticky_mask_generation.glsl")
    sand_transport_compute = compute_program_pipeline(shader_path/"sand_transport_compute.glsl")
    sand_cascade_compute = compute_program_pipeline(shader_path/"sand_cascade_compute.glsl")

    # --- SETUP VBO/VAO ---
    sand_render = RenderingInstance()
    sand_render.setup_vbo_buffer_data(cube_data["position_normals"].nbytes, cube_data["position_normals"], GL.GL_STATIC_DRAW)
    sand_render.setup_vbo_attribs([0, 1], [3, 3], [GL.GL_FLOAT, GL.GL_FLOAT], [24, 24], [0, 12])
    sand_render.setup_ibo_buffer_data(cube_data["indices"].nbytes, cube_data["indices"], GL.GL_STATIC_DRAW)

    # --- CREACIÓN INICIAL DE SSBOs ---
    # Usamos los datos globales generados al inicio
    global_positions_ssbo = SSBO(model_matrices_init, model_matrices_init.nbytes, GL.GL_STATIC_DRAW)
    sand_ssbo = SSBO(sand_slabs_init, sand_slabs_init.nbytes, GL.GL_DYNAMIC_DRAW)
    bedrock_ssbo = SSBO(bedrock_slabs_init, bedrock_slabs_init.nbytes, GL.GL_DYNAMIC_DRAW)
    obstacles_ssbo = SSBO(obstacles_data_init, obstacles_data_init.nbytes, GL.GL_STATIC_DRAW)
    
    # SSBOs vacíos (intermedios)
    empty_bytes = N * N * 2 * 4 # vec2 * float size
    wind_heightfield_ssbo = SSBO(None, empty_bytes, GL.GL_DYNAMIC_DRAW)
    wind_field_ssbo = SSBO(None, empty_bytes, GL.GL_DYNAMIC_DRAW)
    wind_shadowing_ssbo = SSBO(None, N*N*4, GL.GL_DYNAMIC_DRAW)
    sticky_mask_ssbo = SSBO(None, N*N*4, GL.GL_DYNAMIC_DRAW)
    erosion_mask_ssbo = SSBO(None, N*N*4, GL.GL_DYNAMIC_DRAW)

    # === SOLUCIÓN 2: FUNCIÓN DE REINICIO ===
    def reiniciar_simulacion():
        print("Reiniciando simulación...")
        # 1. Regenerar datos en CPU
        _, new_sand, new_bedrock, new_obstacles = generar_datos_iniciales()
        
        # 2. Subir a GPU (Reutilizando los SSBOs existentes)
        # Nota: Asumimos que tu clase SSBO tiene un método setup_SSBO que hace glBufferData
        sand_ssbo.setup_SSBO(new_sand, new_sand.nbytes, GL.GL_DYNAMIC_DRAW)
        bedrock_ssbo.setup_SSBO(new_bedrock, new_bedrock.nbytes, GL.GL_DYNAMIC_DRAW)
        obstacles_ssbo.setup_SSBO(new_obstacles, new_obstacles.nbytes, GL.GL_STATIC_DRAW)
        
        # 3. Limpiar buffers intermedios (Viento, etc) con ceros
        # Esto es importante para borrar el "viento viejo"
        zeros_vec2 = np.zeros(N*N*2, dtype=np.float32)
        zeros_float = np.zeros(N*N, dtype=np.float32)
        
        wind_heightfield_ssbo.setup_SSBO(zeros_vec2, zeros_vec2.nbytes, GL.GL_DYNAMIC_DRAW)
        wind_field_ssbo.setup_SSBO(zeros_vec2, zeros_vec2.nbytes, GL.GL_DYNAMIC_DRAW)
        # ... puedes limpiar los demás si quieres, aunque se sobrescriben en cada frame

    def instanceAttributes(positions, sand, bedrock, obstacle):
        if positions: setInstanceArrayAttribute(global_positions_ssbo.get_SSBO_id(), 2, 2, GL.GL_FLOAT, 8, 1)
        if sand:      setInstanceArrayIAttribute(sand_ssbo.get_SSBO_id(), 3, 1, GL.GL_UNSIGNED_INT, 4, 1)
        if bedrock:   setInstanceArrayIAttribute(bedrock_ssbo.get_SSBO_id(), 4, 1, GL.GL_UNSIGNED_INT, 4, 1)
        if obstacle:  setInstanceArrayIAttribute(obstacles_ssbo.get_SSBO_id(), 5, 1, GL.GL_UNSIGNED_INT, 4, 1)

    fps_display = pyglet.window.FPSDisplay(window=window)
    fps_display.label.font_size = 24
    fps_display.label.color = (255, 0, 0, 255)
    
    gridBlocksX = (N + group_size_x - 1)//group_size_x
    gridBlocksY = (N + group_size_y - 1)//group_size_y
    start_time = time.time() % 1000
    accumulated_time = start_time
    @window.event
    def on_draw():
        global cascade_iterations, transfer_rate, repose_angle, max_steps, sand_transport_block_count, kb, seconds, accumulated_time, time_to_compute_executions
        
        imgui.new_frame()
        this_frame_sec = int(time.time() % 1000 - start_time)

        GL.glClearColor(0.5, 0.5, 0.5, 1.0)
        GL.glLineWidth(1.0)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        window.clear()

        # --- UPDATE COMPUTE SHADERS ---
        # Ejecutamos lógica solo si ha pasado tiempo o en cada frame
        # Para suavidad, lo ejecutamos siempre
        if (time.time() % 1000 - accumulated_time) > time_to_compute_executions: 
            accumulated_time = time.time() % 1000
            # 1. Wind Heightfield
            wind_heightfield_compute.use()
            wind_heightfield_compute["N"] = N
            bedrock_ssbo.bind_SSBO_to_position(0)
            sand_ssbo.bind_SSBO_to_position(1)
            wind_heightfield_ssbo.bind_SSBO_to_position(2)
            wind_heightfield_compute.dispatch(gridBlocksX, gridBlocksY, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
            
            # 2. Wind Field
            wind_update_compute.use()
            wind_update_compute["N"] = N
            wind_update_compute["R_s"] = max_steps
            bedrock_ssbo.bind_SSBO_to_position(0)
            sand_ssbo.bind_SSBO_to_position(1)
            wind_heightfield_ssbo.bind_SSBO_to_position(2)
            wind_field_ssbo.bind_SSBO_to_position(3)
            wind_shadowing_ssbo.bind_SSBO_to_position(4)
            wind_update_compute.dispatch(gridBlocksX, gridBlocksY, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
            
            # 3. Sticky Mask
            sticky_mask_compute.use()
            sticky_mask_compute["N"] = N
            sticky_mask_compute["R_s"] = max_steps
            sticky_mask_compute["cell_size_m"] = cell_size_m
            sticky_mask_compute["h_max"] = h_max
            sticky_mask_compute["kb"] = kb
            sticky_mask_compute["slope_deg_thresh"] = slope_deg_thresh
            bedrock_ssbo.bind_SSBO_to_position(0)
            sand_ssbo.bind_SSBO_to_position(1)
            wind_field_ssbo.bind_SSBO_to_position(3)
            sticky_mask_ssbo.bind_SSBO_to_position(5)
            erosion_mask_ssbo.bind_SSBO_to_position(6)
            sticky_mask_compute.dispatch(gridBlocksX, gridBlocksY, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
            
            # 4. Sand Transport
            sand_transport_compute.use()
            sand_transport_compute["sand_transport_block_count"] = sand_transport_block_count
            sand_transport_compute["N"] = N
            sand_transport_compute["R_s"] = max_steps
            sand_transport_compute["cell_size_m"] = cell_size_m
            bedrock_ssbo.bind_SSBO_to_position(0)
            sand_ssbo.bind_SSBO_to_position(1)
            wind_heightfield_ssbo.bind_SSBO_to_position(2)
            wind_field_ssbo.bind_SSBO_to_position(3)
            wind_shadowing_ssbo.bind_SSBO_to_position(4)
            sticky_mask_ssbo.bind_SSBO_to_position(5)
            erosion_mask_ssbo.bind_SSBO_to_position(6)
            obstacles_ssbo.bind_SSBO_to_position(7)
            sand_transport_compute.dispatch(gridBlocksX, gridBlocksY, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
            
            # 5. Sand Cascade (Gravedad)
            sand_cascade_compute.use()
            sand_cascade_compute["N"] = N
            sand_cascade_compute["cell_size_m"] = cell_size_m
            sand_cascade_compute["tan_repose_angle"] = float(np.tan(np.radians(repose_angle)))
            sand_cascade_compute["transfer_rate"] = transfer_rate
            bedrock_ssbo.bind_SSBO_to_position(0)
            sand_ssbo.bind_SSBO_to_position(1)
            obstacles_ssbo.bind_SSBO_to_position(7)
            for _ in range(cascade_iterations):
                sand_cascade_compute.dispatch(gridBlocksX, gridBlocksY, 1)
                GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

        seconds = this_frame_sec
        
        # --- RENDERING ---
        GL.glEnable(GL.GL_DEPTH_TEST)

        # Sand
        sand_slabs_pipeline.use()
        sand_slabs_pipeline["lightDir"] = lightDir
        sand_slabs_pipeline["lightColor"] = lightColor
        sand_slabs_pipeline["topheight"] = top_sand_height
        setCameraUniforms(camera, sand_slabs_pipeline)
        sand_render.bind_all()
        instanceAttributes(True, True, True, True)
        GL.glDrawElementsInstanced(GL.GL_TRIANGLES, len(cube_data['indices']), GL.GL_UNSIGNED_INT, None, N*N)
        sand_render.unbind_all()
        
        # Bedrock
        bedrock_pipeline.use()
        bedrock_pipeline["lightDir"] = lightDir
        bedrock_pipeline["lightColor"] = lightColor
        setCameraUniforms(camera, bedrock_pipeline)
        sand_render.bind_all()
        instanceAttributes(True, False, True, False)
        GL.glDrawElementsInstanced(GL.GL_TRIANGLES, len(cube_data['indices']), GL.GL_UNSIGNED_INT, None, N*N)
        sand_render.unbind_all()

        # --- IMGUI INTERFACE ---
        imgui.begin("Panel de Control - Sand Sim")
        imgui.separator()

        imgui.text("Configuración de Avalancha")
        _, transfer_rate = imgui.slider_float("Velocidad Caída", transfer_rate, 0.0, 1.0)
        _, repose_angle = imgui.slider_float("Angulo Reposo", repose_angle, 10.0, 89.0)
        _, kb = imgui.slider_float("Kb (Sticky)", kb, 0.0, 1.0)
        _, cascade_iterations = imgui.slider_int("Iteraciones", cascade_iterations, 1, 50)

        imgui.separator()
        imgui.text("Transporte Eólico")
        _, max_steps = imgui.slider_int("Pasos (R_s)", max_steps, 1, 50)
        _, sand_transport_block_count = imgui.slider_int("Bloques/Frame", sand_transport_block_count, 1, 10)

        imgui.separator()
        # BOTÓN DE REINICIO CONECTADO
        _, time_to_compute_executions = imgui.slider_float("Tiempo entre ejecuciones", time_to_compute_executions, 0.0, 1.0)
        if imgui.button("Imprimir Sand SSBO"):
             sand_ssbo.print_content((N, N), np.uint32, label="Arena")
        if imgui.button("Reiniciar Simulación"):
             reiniciar_simulacion()

        imgui.end()

        imgui.render()
        impl.render(imgui.get_draw_data())
        fps_display.draw()

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        if buttons & mouse.RIGHT:
            camera.on_mouse(x, y)

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        camera.on_scroll(scroll_y) 

    def update(dt):
        camera.on_render(dt)
        camera.on_keyboard(keys, dt)

    pyglet.clock.schedule_interval(update, 1/60.0)
    pyglet.app.run()

if __name__ == "__main__":
    sand_move()