from OpenGL import GL
import numpy as np
import ctypes

class SSBO:
    def __init__(self):
        self.id = GL.GLuint(0)
        GL.glGenBuffers(1, self.id)
        print("Generated SSBO with id", self.id)

    def __init__(self, data, n_bytes, usage):
        self.id = GL.GLuint(0)

        GL.glGenBuffers(1, self.id)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.id)
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, n_bytes, data, usage)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        print("Generated SSBO with id", self.id)
    
    def __init__(self, array : np.array, usage):
        self.id = GL.GLuint(0)

        GL.glGenBuffers(1, self.id)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.id)
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, array.nbytes, array, usage)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        print("Generated SSBO with id", self.id)

    def setup_SSBO(self, data, n_bytes, usage):
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.id)
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, n_bytes, data, usage)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

    def bind_SSBO(self):
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.id)

    def bind_SSBO_to_position(self, position):
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, position, self.id)
    
    def unbind_SSBO(self):
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)

    def get_SSBO_id(self):
        return self.id
    


def setInstanceArrayAttribute(buffer_id, position, n_values, type, n_bytes):
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_id)
    GL.glEnableVertexAttribArray(position)
    GL.glVertexAttribPointer(position, n_values, type, GL.GL_FALSE, n_bytes, ctypes.c_void_p(0))
    GL.glVertexAttribDivisor(position, 1)

def setInstanceArrayIAttribute(buffer_id, position, n_values, type, n_bytes):
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_id)
    GL.glEnableVertexAttribArray(position)
    GL.glVertexAttribIPointer(position, n_values, type, n_bytes, ctypes.c_void_p(0))
    GL.glVertexAttribDivisor(position, 1)