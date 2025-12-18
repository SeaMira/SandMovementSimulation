import pyglet
from pyglet.window import key
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
# from grafica.utils import load_pipeline

w_width = 1920
w_height = 1080

group_size_x = 32
group_size_y = 32

N = 512  # número de cubos por lado

top_sand_height = 12
top_bedrock_height = 12

max_steps = 10 # R_s
cell_size_m = 1.0
h_max = 24
kb = 0.1
slope_deg_thresh = 55.0
sand_transport_block_count = 2
repose_angle = 33.0
transfer_rate = 0.25
cascade_iterations = 10

## light settings
lightDir = np.array([1.0, -1.0, 0.0], dtype=np.float32) * (1.0/(2.0**(1.0/2.0)))
lightColor = np.array([1.0, 1.0, 1.0], dtype=np.float32)

sand_heights = generar_alturas(N, scale=0.05, octaves=3, persistence=0.3, lacunarity=1.0, base=0, top_height=top_sand_height, tolerance = -0.5)
bedrock_heights = generar_alturas(N, scale=0.15, octaves=3, persistence=0.7, lacunarity=1.0, base=1, top_height=top_bedrock_height, tolerance=-0.5)
obstacles_data = generar_obstaculos(N, scale=0.02, threshold=0.15, seed=42)
model_matrices = []
sand_slabs = []
bedrock_slabs = []


def setCameraUniforms(c_camera, pipeline):

    pipeline["camPos"] = c_camera.get_pos()

    # le pasamos las matrices al shader
    pipeline["projection"] = c_camera.get_perspective().reshape(
        16, 1, order="F"
    )
    pipeline["view"] = c_camera.get_view().reshape(
        16, 1, order="F"
    )
    pipeline["model"] = c_camera.get_model().reshape(
        16, 1, order="F"
    )

for i in range(N):
    for j in range(N):
        x = i - N/2
        z = j - N/2
        n_sand_slab = int(sand_heights[i, j])  # altura suave
        n_bedrock_slab = int(bedrock_heights[i, j])

        model_matrices.append([x, z])  # <--- lista de 2 elementos por instancia
        sand_slabs.append(n_sand_slab)
        bedrock_slabs.append(n_bedrock_slab)

model_matrices = np.array(model_matrices, dtype=np.float32)
n_instances = len(model_matrices)

sand_slabs = np.array(sand_slabs, dtype=np.uint32)
bedrock_slabs = np.array(bedrock_slabs, dtype=np.uint32)

config = pyglet.gl.Config(
    major_version=4, 
    minor_version=5,   # Tu RTX 2070 soporta hasta 4.6, pero 4.5 es un estándar sólido
    depth_size=24,     # Aseguramos buffer de profundidad para 3D
    double_buffer=True # Doble buffer para evitar parpadeos
)

window = pyglet.window.Window(width=w_width, height=w_height, fullscreen=True, caption="Sand Dunes Simulations", resizable=True, config=config)
keys = key.KeyStateHandler()
window.push_handlers(keys)

camera = cam.Camera(w_width, w_height)
seconds = 0

@click.command("sand_move", short_help="Ejecucion de simulación de movimiento de arena")
def sand_move():
    global model_matrices, N, n_instances, sand_slabs, bedrock_slabs, seconds

    # --- AGREGA ESTO PARA VERIFICAR LA GPU ---
    print("-" * 30)
    print("REPORTE DE GPU:")
    vendor = GL.glGetString(GL.GL_VENDOR).decode()
    renderer = GL.glGetString(GL.GL_RENDERER).decode()
    version = GL.glGetString(GL.GL_VERSION).decode()
    print(f"Vendor:   {vendor}")
    print(f"Renderer: {renderer}") # <--- AQUÍ DEBE DECIR NVIDIA / AMD
    print(f"OpenGL:   {version}")
    print("-" * 30)
    
    if "Intel" in renderer or "Microsoft" in renderer:
        print("⚠️ PRECAUCIÓN: Estás usando la GPU Integrada. El rendimiento será bajo.")
        print("Configura 'Alto Rendimiento' en Windows para python.exe")

    # primer elemento: el rectángulo de fondo
    cube_data = cubo_unitario()

    # reusamos nuestros shaders
    sand_slabs_pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "shaders" / "sand_vs.glsl", 
        Path(os.path.dirname(__file__)) / "shaders" / "sand_fs.glsl") 
    bedrock_pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "shaders" / "bedrock_vs.glsl", 
        Path(os.path.dirname(__file__)) / "shaders" / "bedrock_fs.glsl") 


    sand_render = RenderingInstance()
    sand_render.setup_vbo_buffer_data(cube_data["position_normals"].nbytes, cube_data["position_normals"], GL.GL_STATIC_DRAW)
    sand_render.setup_vbo_attribs(
        [0, 1],
        [3, 3],
        [GL.GL_FLOAT, GL.GL_FLOAT],
        [24, 24],
        [0, 12]
    )
    sand_render.setup_ibo_buffer_data(cube_data["indices"].nbytes, cube_data["indices"], GL.GL_STATIC_DRAW)

    # ssbo para posiciones
    global_positions_ssbo = SSBO(model_matrices, model_matrices.nbytes, GL.GL_STATIC_DRAW)

    # ssbo con contador de slabs de arena
    sand_ssbo = SSBO(sand_slabs, sand_slabs.nbytes, GL.GL_DYNAMIC_DRAW)
    
    # ssbo con contador de slabs de bedrock
    bedrock_ssbo = SSBO(bedrock_slabs, bedrock_slabs.nbytes, GL.GL_DYNAMIC_DRAW)
    
    # ssbo con wind heightfield
    wind_heightfield_ssbo = SSBO(None, (N*N*2*4), GL.GL_DYNAMIC_DRAW)
    
    # ssbo con wind field
    wind_field_ssbo = SSBO(None, (N*N*2*4), GL.GL_DYNAMIC_DRAW)
    
    # ssbo con wind shadowing
    wind_shadowing_ssbo = SSBO(None, (N*N*4), GL.GL_DYNAMIC_DRAW)
    
    # ssbo con sticky mask
    sticky_mask_ssbo = SSBO(None, (N*N*4), GL.GL_DYNAMIC_DRAW)
    
    # ssbo con erosion mask
    erosion_mask_ssbo = SSBO(None, (N*N*4), GL.GL_DYNAMIC_DRAW)
    
    # ssbo con obstacles
    obstacles_ssbo = SSBO(obstacles_data, obstacles_data.nbytes, GL.GL_STATIC_DRAW)
    
    shader_path = Path(os.path.dirname(__file__)) / "shaders"
    print(shader_path)

    wind_heightfield_compute = compute_program_pipeline(shader_path/"wind_heightfield_compute.glsl")
    wind_update_compute = compute_program_pipeline(shader_path/"wind_update_compute.glsl")
    sticky_mask_compute = compute_program_pipeline(shader_path/"sticky_mask_generation.glsl")
    sand_transport_compute = compute_program_pipeline(shader_path/"sand_transport_compute.glsl")
    sand_cascade_compute = compute_program_pipeline(shader_path/"sand_cascade_compute.glsl")
    
    
    
    def instanceAttributes(positions_ssbo_bool, sand_ssbo_bool, bedrock_ssbo_bool, obstacle_ssbo_bool):
        if positions_ssbo_bool:
            setInstanceArrayAttribute(global_positions_ssbo.get_SSBO_id(), 2, 2, GL.GL_FLOAT, 8, 1)
        if sand_ssbo_bool:    
            setInstanceArrayIAttribute(sand_ssbo.get_SSBO_id(), 3, 1, GL.GL_UNSIGNED_INT, 4, 1)
        if bedrock_ssbo_bool:
            setInstanceArrayIAttribute(bedrock_ssbo.get_SSBO_id(), 4, 1, GL.GL_UNSIGNED_INT, 4, 1)
        if obstacle_ssbo_bool:
            setInstanceArrayIAttribute(obstacles_ssbo.get_SSBO_id(), 5, 1, GL.GL_UNSIGNED_INT, 4, 1)
        
    start_time = time.time() % 1000

    fps_display = pyglet.window.FPSDisplay(window=window)
    fps_display.label.font_size = 24  # Opcional: hacerlo más grande
    fps_display.label.color = (255, 0, 0, 255) # Opcional: color rojo para que contraste
    gridBlocksX = (N + group_size_x - 1)//group_size_x
    gridBlocksY = (N + group_size_y - 1)//group_size_y

    @window.event
    def on_draw():
        global seconds
        this_frame_sec = int(time.time() % 1000 - start_time)

        GL.glClearColor(0.5, 0.5, 0.5, 1.0)
        GL.glLineWidth(1.0)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        window.clear()

        if (this_frame_sec - seconds) == 1: 
            ####################
            ## WIND HEIGHTFIELD
            wind_heightfield_compute.use()
            wind_heightfield_compute["N"] = N

            bedrock_ssbo.bind_SSBO_to_position(0)
            sand_ssbo.bind_SSBO_to_position(1)
            wind_heightfield_ssbo.bind_SSBO_to_position(2)
            wind_heightfield_compute.dispatch(gridBlocksX, gridBlocksY, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
            
            ####################
            ## WIND FIELD
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
            
            ####################
            ## STICKY MASK
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
            
            ####################
            ## SAND TRANSPORT
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
            
            ####################
            ## SAND CASCADE
            sand_cascade_compute.use()
            sand_cascade_compute["N"] = N
            sand_cascade_compute["cell_size_m"] = cell_size_m
            sand_cascade_compute["tan_repose_angle"] = np.tan(np.radians(repose_angle))
            sand_cascade_compute["transfer_rate"] = transfer_rate

            bedrock_ssbo.bind_SSBO_to_position(0)
            sand_ssbo.bind_SSBO_to_position(1)
            obstacles_ssbo.bind_SSBO_to_position(7)

            for _ in range(cascade_iterations):
                sand_cascade_compute.dispatch(gridBlocksX, gridBlocksY, 1)
                GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

        seconds = this_frame_sec
        # lo activamos a la hora de graficar nuestra escena
        GL.glEnable(GL.GL_DEPTH_TEST)

        # sand slabs pipeline 
        sand_slabs_pipeline.use()

        # setting uniforms
        sand_slabs_pipeline["lightDir"] = lightDir
        sand_slabs_pipeline["lightColor"] = lightColor
        sand_slabs_pipeline["topheight"] = top_sand_height
        setCameraUniforms(camera, sand_slabs_pipeline)

        # binding vertex buffers
        sand_render.bind_all()

        # setting attributes
        instanceAttributes(True, True, True, True)

        GL.glDrawElementsInstanced(GL.GL_TRIANGLES, len(cube_data['indices']), GL.GL_UNSIGNED_INT, None, N*N)
        sand_render.unbind_all()
        
        
        # bedrock pipeline 
        bedrock_pipeline.use()

        # setting uniforms
        bedrock_pipeline["lightDir"] = lightDir
        bedrock_pipeline["lightColor"] = lightColor
        setCameraUniforms(camera, bedrock_pipeline)

        # binding vertex buffers
        sand_render.bind_all()

        # setting attributes
        instanceAttributes(True, False, True, False)

        GL.glDrawElementsInstanced(GL.GL_TRIANGLES, len(cube_data['indices']), GL.GL_UNSIGNED_INT, None, N*N)
        sand_render.unbind_all()

        fps_display.draw()

        

    @window.event
    def on_mouse_motion(x, y, dx, dy):
        camera.on_mouse(x, y)

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        camera.on_scroll(scroll_y)   # scroll_y positivo = acercar, negativo = alejar

    def update(dt):
        camera.on_render(dt)
        camera.on_keyboard(keys, dt)  # movimiento WASD

    pyglet.clock.schedule_interval(update, 1/60.0)
    pyglet.app.run()

if __name__ == "__main__":
    sand_move()