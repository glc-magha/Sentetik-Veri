"""#Temel Üçgen Çizimi
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np

# Shader kaynakları
vertex_src = """
#version 330
layout(location = 0) in vec3 a_position;
void main() {
    gl_Position = vec4(a_position, 1.0);
}
"""

fragment_src = """
#version 330
out vec4 FragColor;
void main() {
    FragColor = vec4(0.2, 0.6, 1.0, 1.0);
}
"""

# GLFW pencere kurulumu
if not glfw.init():
    raise Exception("GLFW başlatılamadı")

window = glfw.create_window(800, 600, "Üçgen", None, None)
glfw.make_context_current(window)

# Üçgen köşe noktaları
vertices = np.array([
     0.0,  0.5, 0.0,
    -0.5, -0.5, 0.0,
     0.5, -0.5, 0.0
], dtype=np.float32)

shader = compileProgram(
    compileShader(vertex_src, GL_VERTEX_SHADER),
    compileShader(fragment_src, GL_FRAGMENT_SHADER)
)

# Vertex Buffer ve Array tanımlama
VAO = glGenVertexArrays(1)
VBO = glGenBuffers(1)

glBindVertexArray(VAO)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

# Ana döngü
while not glfw.window_should_close(window):
    glfw.poll_events()
    glClearColor(0, 0, 0, 1)
    glClear(GL_COLOR_BUFFER_BIT)

    glUseProgram(shader)
    glBindVertexArray(VAO)
    glDrawArrays(GL_TRIANGLES, 0, 3)

    glfw.swap_buffers(window)

glfw.terminate()
#-----------------------------------------------------------
#Dönen 3D Küp (Renkli, Kamera Yok)
# pyrr ile dönüş matrisi ekleyebilirsin
rotation = pyrr.Matrix44.from_y_rotation(time.time() % 6.28)
#-------------------------------------------------------------
pip install glfw PyOpenGL pyrr


"""