from pyglet.window import key, mouse
from pyglet import gl
import utils.transformations as tr
import math
import numpy as np

class Camera:
    def __init__(self, scr_width=800, scr_height=600, pos=np.array([0.0, 0.0, 0.0]), front=np.array([0.0, 0.0, 1.0]), up=np.array([0.0, 1.0, 0.0])):
        self.pos = pos
        self.front = front # front vector of the camera.
        self.up    = up # up vector of the camera.
        self.right    = np.array([-1.0, 0.0,  0.0]) # right vector of the camera.
        self.initUp    = np.array([0.0, 1.0,  0.0]) # initial up vector of the camera.

        self.yaw, self.pitch = 180.0, 0.0  # rotaciones
        self.lastX, self.lastY = 400, 300
        self.first_mouse = True
        self.fov = 45.0
        self.scr_width = scr_width
        self.scr_height = scr_height
        self.speed = 10.0
        self.near = 0.1
        self.far = 100.0

        self.margin = 20.0
        self.edge_step = 70.0

        self.OnUpperEdge = False # whether the mouse is on the upper edge of the screen.
        self.OnLowerEdge = False # whether the mouse is on the lower edge of the screen.
        self.OnLeftEdge = False # whether the mouse is on the left edge of the screen.
        self.OnRightEdge = False # whether the mouse is on the right edge of the screen.


    def get_pos(self):
        return self.pos
    def get_front(self):
        return self.front
    def get_up(self):
        return self.up
    def get_right(self):
        return self.right
    
    def get_perspective(self):
        return tr.perspective(self.fov, self.scr_width/self.scr_height, self.near, self.far)
    
    def get_orthographic(self):
        return tr.ortho(-self.scr_width/200, self.scr_width/200, -self.scr_height/200, self.scr_height/200, self.near, self.far)
    
    def get_view(self):
        return tr.lookAt(self.pos, self.pos + self.front, self.up)
    
    def get_model(self):
        return tr.identity()
    
    def get_fov(self):
        return self.fov
    def get_pitch(self):
        return self.pitch
    def get_yaw(self):
        return self.yaw
    def get_near(self):
        return self.near
    def get_far(self):
        return self.far
    
    def set_pos(self, pos):
        self.pos = pos
    def set_front(self, front):
        self.front = front
    def set_up(self, up):
        self.up = up

    def set_margin(self, margin):
        self.margin = margin
    def set_edge_step(self, edge_step):
        self.edge_step = edge_step
    def set_scr_size(self, width, height):
        self.scr_width = width
        self.scr_height = height

    def look_at(self, point):
        direction = point - self.pos
        direction = direction / np.linalg.norm(direction)
        self.front = direction

        self.yaw = math.degrees(math.atan2(direction[2], -direction[0]))
        self.pitch = math.degrees(math.asin(direction[1]))
        self.update()
    

    def get_direction(self):
        """Devuelve vector dirección a partir de yaw/pitch"""
        dx = math.cos(math.radians(self.pitch)) * math.sin(math.radians(self.yaw))
        dy = math.sin(math.radians(self.pitch))
        dz = -math.cos(math.radians(self.pitch)) * math.cos(math.radians(self.yaw))
        return dx, dy, dz

    def on_keyboard(self, keys, dt):
        """
        keys: instancia de KeyStateHandler de pyglet
        dt: delta time en segundos
        """
        cameraSpeed = self.speed * dt

        # Adelante/atrás
        if keys[key.W]:
            self.pos += self.front * cameraSpeed
        if keys[key.S]:
            self.pos -= self.front * cameraSpeed

        # Izquierda/derecha
        if keys[key.A]:
            self.pos -= self.right * cameraSpeed
        if keys[key.D]:
            self.pos += self.right * cameraSpeed

        # Arriba/abajo
        if keys[key.SPACE]:
            self.pos += self.initUp * cameraSpeed
        if keys[key.LSHIFT]:
            self.pos -= self.initUp * cameraSpeed

        # Ajustar velocidad
        if keys[key.E]:
            self.speed += 1.0
        if keys[key.Q]:
            self.speed = max(1.0, self.speed - 1.0)
        
        if keys[key.F]:
            self.show_info()

    def on_mouse(self, xpos, ypos):
        # calcular desplazamiento
        xoffset = xpos - self.lastX
        yoffset = ypos - self.lastY
        self.lastX = xpos
        self.lastY = ypos

        sensitivity = 0.1
        xoffset *= sensitivity
        yoffset *= sensitivity

        self.yaw += xoffset
        self.pitch += yoffset

        # limitar pitch
        if self.pitch > 89.0:
            self.pitch = 89.0
        if self.pitch < -89.0:
            self.pitch = -89.0

        # detección de bordes de pantalla
        if abs(xoffset) <= self.margin and xpos <= self.margin:
            self.OnLeftEdge, self.OnRightEdge = True, False
        elif abs(xoffset) <= self.margin and xpos >= (self.scr_width - self.margin):
            self.OnLeftEdge, self.OnRightEdge = False, True
        else:
            self.OnLeftEdge, self.OnRightEdge = False, False

        if abs(yoffset) <= self.margin and ypos <= self.margin:
            self.OnUpperEdge, self.OnLowerEdge = True, False
        elif abs(yoffset) <= self.margin and ypos >= (self.scr_height - self.margin):
            self.OnUpperEdge, self.OnLowerEdge = False, True
        else:
            self.OnUpperEdge, self.OnLowerEdge = False, False

        self.update()  # recalcula front, right, up

    def on_render(self, dt):
        should_update = False

        if self.OnLeftEdge:
            self.yaw -= self.edge_step * dt
            should_update = True
        elif self.OnRightEdge:
            self.yaw += self.edge_step * dt
            should_update = True

        if self.OnUpperEdge:
            if self.pitch > -90.0:
                self.pitch -= self.edge_step * dt
                should_update = True
        elif self.OnLowerEdge:
            if self.pitch < 90.0:
                self.pitch += self.edge_step * dt
                should_update = True

        if should_update:
            # limitar pitch
            self.pitch = max(-89.0, min(89.0, self.pitch))
            self.update()

    def on_scroll(self, yoffset):
        """Ajusta el FOV con la rueda del mouse"""
        self.fov -= yoffset
        if self.fov < 1.0:
            self.fov = 1.0
        if self.fov > 180.0:
            self.fov = 180.0


    def update(self):
        """Recalcula los vectores de la cámara (front, right, up) a partir de yaw y pitch"""
        # Vector frente
        front = np.array([
            math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch)),
            math.sin(math.radians(self.pitch)),
            math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        ], dtype=np.float32)

        # Normalizar
        self.front = front / np.linalg.norm(front)

        # Recalcular right y up
        self.right = np.cross(self.front, self.initUp)
        self.right /= np.linalg.norm(self.right)

        self.up = np.cross(self.right, self.front)
        self.up /= np.linalg.norm(self.up)


    def show_info(self):
        """Imprime información de la cámara"""
        print("=== Camera Info ===")
        print(f"Position: {self.pos[0]:.2f}, {self.pos[1]:.2f}, {self.pos[2]:.2f}")
        print(f"Front:    {self.front[0]:.2f}, {self.front[1]:.2f}, {self.front[2]:.2f}")
        print(f"Up:       {self.up[0]:.2f}, {self.up[1]:.2f}, {self.up[2]:.2f}")
        print(f"Right:    {self.right[0]:.2f}, {self.right[1]:.2f}, {self.right[2]:.2f}")
        print(f"Yaw: {self.yaw:.2f}, Pitch: {self.pitch:.2f}, FOV: {self.fov:.2f}")

