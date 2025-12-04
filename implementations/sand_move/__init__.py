import pyglet
from pyglet.window import key
import implementations.sand_move.camera as cam
from OpenGL import GL
import numpy as np
import os
from pathlib import Path
from utils.load_pipeline import load_pipeline, compute_program_pipeline
from utils.elementos import rectangulo, cubo_unitario
from utils.gl_utils import SSBO, setInstanceArrayAttribute, setInstanceArrayIAttribute
from implementations.sand_move.height_map_noise import generar_alturas
import click
import ctypes
import time
# from grafica.utils import load_pipeline

w_width = 800
w_height = 600

group_size = 64

N = 100  # número de cubos por lado

## light settings
lightPos = np.array([N/2, N, N/2], dtype=np.float32)
lightColor = np.array([1.0, 1.0, 1.0], dtype=np.float32)

alturas = generar_alturas(N, scale=0.15, octaves=3, persistence=0.7, lacunarity=1.0, base=0, top_height=6)
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
        n_sand_slab = int(alturas[i, j]) + 1  # altura suave
        n_bedrock_slab = np.random.randint(1, 3)

        model_matrices.append([x, z])  # <--- lista de 2 elementos por instancia
        sand_slabs.append(n_sand_slab)
        bedrock_slabs.append(n_bedrock_slab)

model_matrices = np.array(model_matrices, dtype=np.float32)
n_instances = len(model_matrices)

sand_slabs = np.array(sand_slabs, dtype=np.uint32)
bedrock_slabs = np.array(bedrock_slabs, dtype=np.uint32)

window = pyglet.window.Window(width=w_width, height=w_height, caption="Camara", resizable=True)
keys = key.KeyStateHandler()
window.push_handlers(keys)

camera = cam.Camera()

@click.command("sand_move", short_help="Ejecucion de simulación de movimiento de arena")
def sand_move():
    global model_matrices, N, n_instances, sand_slabs, bedrock_slabs
    # primer elemento: el rectángulo de fondo
    cube_data = cubo_unitario()

    # reusamos nuestros shaders
    bg_pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "shaders" / "simple_vs.glsl", 
        Path(os.path.dirname(__file__)) / "shaders" / "simple_fs.glsl") 

    vao = GL.GLuint(0)
    vbo = GL.GLuint(0)
    ibo = GL.GLuint(0)
    GL.glGenVertexArrays(1, vao)
    GL.glBindVertexArray(vao)

    GL.glGenBuffers(1, vbo)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, cube_data["position_normals"].nbytes, cube_data["position_normals"], GL.GL_STATIC_DRAW)
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, ctypes.c_void_p(0))
    GL.glEnableVertexAttribArray(1)
    GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, ctypes.c_void_p(12))
    
    GL.glGenBuffers(1, ibo)
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ibo)
    GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, cube_data["indices"].nbytes, cube_data["indices"], GL.GL_STATIC_DRAW)

    # ssbo para posiciones
    global_positions_ssbo = SSBO(model_matrices, GL.GL_STATIC_DRAW)

    # ssbo con contador de slabs de arena
    sand_ssbo = SSBO(sand_slabs, GL.GL_DYNAMIC_DRAW)
    
    # ssbo con contador de slabs de bedrock
    bedrock_ssbo = SSBO(bedrock_slabs, GL.GL_DYNAMIC_DRAW)
    
    compute_pipeline = compute_program_pipeline(
        Path(os.path.dirname(__file__)) / "shaders" / "simple_compute.glsl"
    )
    
    def instanceAttributes():
        setInstanceArrayAttribute(global_positions_ssbo.get_SSBO_id(), 2, 2, GL.GL_FLOAT, 8)
        
        setInstanceArrayIAttribute(sand_ssbo.get_SSBO_id(), 3, 1, GL.GL_UNSIGNED_INT, 4)

        setInstanceArrayIAttribute(bedrock_ssbo.get_SSBO_id(), 4, 1, GL.GL_UNSIGNED_INT, 4)
        

    @window.event
    def on_draw():
        GL.glClearColor(0.5, 0.5, 0.5, 1.0)
        GL.glLineWidth(1.0)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        window.clear()

        compute_pipeline.use()
        # compute_pipeline["time"] = time.time() % 1000
         # vinculamos el SSBO al binding 0
        global_positions_ssbo.bind_SSBO_to_position(0)
        compute_pipeline.dispatch((n_instances + group_size - 1)//group_size, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

        # lo activamos a la hora de graficar nuestra escena
        GL.glEnable(GL.GL_DEPTH_TEST)
        bg_pipeline.use()

        bg_pipeline["lightPos"] = lightPos
        bg_pipeline["lightColor"] = lightColor
        setCameraUniforms(camera, bg_pipeline)

        GL.glBindVertexArray(vao)
        instanceAttributes()

        GL.glDrawElementsInstanced(GL.GL_TRIANGLES, len(cube_data['indices']), GL.GL_UNSIGNED_INT, None, N*N)

        

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