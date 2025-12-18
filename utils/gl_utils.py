from OpenGL import GL
import numpy as np
import ctypes

def setInstanceArrayAttribute(buffer_id, position, n_values, type, n_bytes, divisor, offset = 0):
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_id)
    GL.glEnableVertexAttribArray(position)
    GL.glVertexAttribPointer(position, n_values, type, GL.GL_FALSE, n_bytes, ctypes.c_void_p( offset ))
    GL.glVertexAttribDivisor(position, divisor)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

def setInstanceArrayIAttribute(buffer_id, position, n_values, type, n_bytes, divisor, offset = 0):
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_id)
    GL.glEnableVertexAttribArray(position)
    GL.glVertexAttribIPointer(position, n_values, type, n_bytes, ctypes.c_void_p( offset ))
    GL.glVertexAttribDivisor(position, divisor)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)


class SSBO:

    def __init__(self, data=None, n_bytes=None, usage=None):
        self.id = GL.glGenBuffers(1)
        if n_bytes and usage:
            self.setup_SSBO(data, n_bytes, usage)

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
    
    def read_data(self, shape, dtype):
        """
        Lee el contenido del buffer desde la GPU.
        
        Args:
            shape (tuple): La forma del array resultante (ej. (512, 512) o (N*N,)).
            dtype (type): El tipo de dato de numpy (ej. np.uint32, np.float32).
            
        Returns:
            np.array: Un array de numpy con los datos leídos.
        """
        self.bind_SSBO()
        
        # Mapeamos el buffer de GPU a un puntero en memoria CPU (solo lectura)
        ptr = GL.glMapBuffer(GL.GL_SHADER_STORAGE_BUFFER, GL.GL_READ_ONLY)
        
        if not ptr:
            print(f"Error: No se pudo mapear el SSBO {self.id}")
            self.unbind_SSBO()
            return np.zeros(shape, dtype=dtype)

        # Calculamos el tamaño total en bytes
        total_bytes = np.prod(shape) * np.dtype(dtype).itemsize
        
        # Copiamos los datos crudos a un array de numpy
        try:
            data = np.frombuffer(ctypes.string_at(ptr, size=total_bytes), dtype=dtype)
        except Exception as e:
            print(f"Error leyendo SSBO: {e}")
            data = np.zeros(np.prod(shape), dtype=dtype)
        
        # Desmapeamos y liberamos
        GL.glUnmapBuffer(GL.GL_SHADER_STORAGE_BUFFER)
        self.unbind_SSBO()
        
        return data.reshape(shape)

    def print_content(self, shape, dtype, label="SSBO Data", n_print=10):
        """
        Imprime en consola un resumen del contenido del buffer.
        Util para debug rápido.
        """
        data = self.read_data(shape, dtype)
        flat_data = data.flatten()
        
        print(f"--- {label} ---")
        print(f"Min: {np.min(data)}, Max: {np.max(data)}, Sum: {np.sum(data)}")
        print(f"Primeros {n_print} valores: {flat_data[:n_print]}")
        print("--------------------")
    

class RenderingInstance:
    def __init__(self):
        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)
        self.ibo = GL.glGenBuffers(1)

        print("Generated VAO with id", self.vao)
        print("Generated VBO with id", self.vbo)
        print("Generated IBO with id", self.ibo)

    def bind_vao(self):
        GL.glBindVertexArray(self.vao)

    def bind_vbo(self):
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
    
    def bind_ibo(self):
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ibo)
    
    def bind_all(self):
        self.bind_vao()
        self.bind_vbo()
        self.bind_ibo()
    
    def unbind_all(self):
        self.unbind_vao()
        self.unbind_vbo()
        self.unbind_ibo()

    def unbind_vao(self):
        GL.glBindVertexArray(0)

    def unbind_vbo(self):
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
    
    def unbind_ibo(self):
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    def setup_vbo_buffer_data(self, nbytes, data, usage):
        self.bind_vao()
        self.bind_vbo()

        GL.glBufferData(GL.GL_ARRAY_BUFFER, nbytes, data, usage)
        
        self.unbind_vbo()
        self.unbind_vao()
    
    def setup_vbo_attribs(self, 
                          attribs_array, 
                          n_values_array, 
                          value_types_array, 
                          positions_skip_array, 
                          offset_array
                          ):
        assert(len(attribs_array) == len(n_values_array))
        assert(len(value_types_array) == len(n_values_array))
        assert(len(value_types_array) == len(positions_skip_array))
        assert(len(offset_array) == len(positions_skip_array))

        self.bind_vao()

        n_attribs = len(attribs_array)
        for i in range(n_attribs):
            t = value_types_array[i]
            if t == GL.GL_FLOAT:
                setInstanceArrayAttribute(self.vbo,
                                          attribs_array[i], 
                                          n_values_array[i],
                                          t,
                                          positions_skip_array[i],
                                          0,
                                          offset_array[i]
                                          )
            if t == GL.GL_UNSIGNED_INT:
                setInstanceArrayIAttribute(self.vbo,
                                          attribs_array[i], 
                                          n_values_array[i],
                                          t,
                                          positions_skip_array[i],
                                          0,
                                          offset_array[i] 
                                          )
        
        self.unbind_vao()

    def setup_ibo_buffer_data(self, nbytes, data, usage):
        self.bind_vao()
        self.bind_vbo()
        self.bind_ibo()

        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, nbytes, data, usage)
        
        self.unbind_ibo()
        self.unbind_vbo()
        self.unbind_vao()

    def get_vao(self):
        return self.vao
        
    def get_vbo(self):
        return self.vbo
    
    def get_ibo(self):
        return self.ibo