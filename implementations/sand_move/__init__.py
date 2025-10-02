import pyglet
from pyglet.window import key
import implementations.sand_move.camera as cam
from OpenGL import GL
import numpy as np
import os
from pathlib import Path
from utils.load_pipeline import load_pipeline, compute_program_pipeline
from utils.elementos import rectangulo, cubo_unitario
import click
import ctypes
import time
# from grafica.utils import load_pipeline

w_width = 800
w_height = 600

group_size = 64

N = 30  # número de cubos por lado

## light settings
lightPos = np.array([N/2, 5.0, N/2], dtype=np.float32)
lightColor = np.array([1.0, 1.0, 1.0], dtype=np.float32)

model_matrices = []
sand_slabs = []
bedrock_slabs = []

for i in range(N):
    for j in range(N):
        x = i - N/2
        z = j - N/2
        n_sand_slab = np.random.randint(1, 6)
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
    ssbo = GL.GLuint(0)
    GL.glGenBuffers(1, ssbo)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, ssbo)
    GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, model_matrices.nbytes, model_matrices, GL.GL_STATIC_DRAW)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

    # ssbo con contador de slabs de arena
    sand_ssbo = GL.GLuint(0)
    GL.glGenBuffers(1, sand_ssbo)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, sand_ssbo)
    GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, sand_slabs.nbytes, sand_slabs, GL.GL_DYNAMIC_DRAW)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
    
    # ssbo con contador de slabs de bedrock
    bedrock_ssbo = GL.GLuint(0)
    GL.glGenBuffers(1, bedrock_ssbo)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, bedrock_ssbo)
    GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, bedrock_slabs.nbytes, bedrock_slabs, GL.GL_DYNAMIC_DRAW)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
    
    compute_pipeline = compute_program_pipeline(
        Path(os.path.dirname(__file__)) / "shaders" / "simple_compute.glsl"
    )
    


    @window.event
    def on_draw():
        GL.glClearColor(0.5, 0.5, 0.5, 1.0)
        GL.glLineWidth(1.0)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        window.clear()

        compute_pipeline.use()
        # compute_pipeline["time"] = time.time() % 1000
         # vinculamos el SSBO al binding 0
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 0, ssbo)
        compute_pipeline.dispatch((n_instances + group_size - 1)//group_size, 1, 1)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

        # lo activamos a la hora de graficar nuestra escena
        GL.glEnable(GL.GL_DEPTH_TEST)
        bg_pipeline.use()

        bg_pipeline["lightPos"] = lightPos
        bg_pipeline["lightColor"] = lightColor
        bg_pipeline["camPos"] = camera.get_pos()

        # le pasamos las matrices al shader
        bg_pipeline["projection"] = camera.get_perspective().reshape(
            16, 1, order="F"
        )
        bg_pipeline["view"] = camera.get_view().reshape(
            16, 1, order="F"
        )
        bg_pipeline["model"] = camera.get_model().reshape(
            16, 1, order="F"
        )
        GL.glBindVertexArray(vao)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, ssbo)
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 2, GL.GL_FLOAT, GL.GL_FALSE, 8, ctypes.c_void_p(0))
        GL.glVertexAttribDivisor(2, 1)
        
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, sand_ssbo)
        GL.glEnableVertexAttribArray(3)
        GL.glVertexAttribIPointer(3, 1, GL.GL_UNSIGNED_INT, 4, ctypes.c_void_p(0))
        GL.glVertexAttribDivisor(3, 1)
        
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, bedrock_ssbo)
        GL.glEnableVertexAttribArray(4)
        GL.glVertexAttribIPointer(4, 1, GL.GL_UNSIGNED_INT, GL.GL_FALSE, 4, ctypes.c_void_p(0))
        GL.glVertexAttribDivisor(4, 1)

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