"""OpenGL (Open Graphics Library), 2D ve 3D grafiklerin çizimi için geliştirilmiş, platform ve dil bağımsız bir grafik API’sidir. Genellikle oyun motorlarında, CAD programlarında, simülasyonlarda ve diğer görsel uygulamalarda kullanılır.

🔹 OpenGL Nedir?
1992’de Silicon Graphics Inc. (SGI) tarafından geliştirildi.

Grafik donanımından bağımsız olarak çalışan, donanıma grafik çizimi komutları gönderen bir arabirimdir.

C ve C++ dilleri ile sıkça kullanılır.

Gerçek zamanlı render (anlık çizim) sağlar.

🧠 OpenGL Ne İşe Yarar?
3D nesneleri modellemek ve çizmek

Işıklandırma, gölgelendirme ve doku kaplama işlemleri

Kamerayla sahne gezdirme (view transformation)

2D oyunlar veya arayüzler çizmek

GPU hızlandırmalı grafik işlemleri yapmak






🧪 Basit OpenGL Uygulama Örneği (C++ ile)
cpp
#include <GL/glut.h>
void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_TRIANGLES);       // Üçgen çizmeye başla
        glColor3f(1.0, 0.0, 0.0);  // Kırmızı renk
        glVertex2f(-0.5, -0.5);    // Nokta 1
        glColor3f(0.0, 1.0, 0.0);  // Yeşil renk
        glVertex2f(0.5, -0.5);     // Nokta 2
        glColor3f(0.0, 0.0, 1.0);  // Mavi renk
        glVertex2f(0.0, 0.5);      // Nokta 3
    glEnd();
    glFlush();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);                     // GLUT başlat
    glutCreateWindow("OpenGL Üçgen");         // Pencere oluştur
    glutDisplayFunc(display);                 // Çizim fonksiyonu ata
    glutMainLoop();                           // Sonsuz döngü
    return 0;
}

📦 OpenGL Kullanım Alanları:
Oyun Geliştirme: Unity veya Unreal dışı motorlarda grafik çizimi için

Bilimsel Simülasyonlar: Moleküler modelleme, fizik simülasyonları

Veri Görselleştirme: 3D grafiklerle veri analizi

Eğitim Amaçlı Projeler: Grafik dersi uygulamaları

GUI Uygulamaları: Qt gibi kütüphanelerle birlikte

🧩 OpenGL Alternatifleri:
DirectX (Windows’a özel)

Vulkan (düşük seviyeli, modern API)

Metal (Apple platformları için)

WebGL (Web tarayıcılarında çalışan OpenGL ES sürümü)"""

#......................................................................................
#Basit Üçgen Çizimi:
import pip


class PyOpenGL_accelerate:
    pass


PyOpenGL_accelerate
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *


def draw():
    glClear(GL_COLOR_BUFFER_BIT)
    glBegin(GL_TRIANGLES)
    glColor3f(1, 0, 0)  # Kırmızı
    glVertex2f(-0.5, -0.5)
    glColor3f(0, 1, 0)  # Yeşil
    glVertex2f(0.5, -0.5)
    glColor3f(0, 0, 1)  # Mavi
    glVertex2f(0.0, 0.5)
    glEnd()
    glFlush()


glutInit()
glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
glutInitWindowSize(500, 500)
glutCreateWindow("PyOpenGL Üçgen")
glutDisplayFunc(draw)
glutMainLoop()
#----------------------------------------------------------------------
#3D Nesne: Dönen Küp (PyOpenGL ile)
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

angle = 0


def draw():
    global angle
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, -5)
    glRotatef(angle, 1, 1, 0)

    glBegin(GL_QUADS)

    # Ön yüz (kırmızı)
    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(1.0, 1.0, -1.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glVertex3f(-1.0, 1.0, 1.0)
    glVertex3f(1.0, 1.0, 1.0)

    # Arka yüz (yeşil)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(1.0, -1.0, 1.0)
    glVertex3f(-1.0, -1.0, 1.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(1.0, -1.0, -1.0)

    # Diğer yüzler...
    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(-1.0, 1.0, 1.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(-1.0, -1.0, 1.0)

    glColor3f(1.0, 1.0, 0.0)
    glVertex3f(1.0, 1.0, -1.0)
    glVertex3f(1.0, 1.0, 1.0)
    glVertex3f(1.0, -1.0, 1.0)
    glVertex3f(1.0, -1.0, -1.0)

    glColor3f(1.0, 0.0, 1.0)
    glVertex3f(1.0, 1.0, 1.0)
    glVertex3f(-1.0, 1.0, 1.0)
    glVertex3f(-1.0, -1.0, 1.0)
    glVertex3f(1.0, -1.0, 1.0)

    glColor3f(0.0, 1.0, 1.0)
    glVertex3f(1.0, -1.0, -1.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glVertex3f(1.0, 1.0, -1.0)

    glEnd()

    glutSwapBuffers()
    angle += 1


def timer(value):
    glutPostRedisplay()
    glutTimerFunc(16, timer, 0)


def init():
    glEnable(GL_DEPTH_TEST)


glutInit()
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
glutInitWindowSize(600, 600)
glutCreateWindow("3D Dönen Küp")
init()
glutDisplayFunc(draw)
glutTimerFunc(0, timer, 0)
glutMainLoop()

#----------------------------------------------------------------
#3D Nesne: Küre (GLU Kütüphanesi ile)
def draw_sphere():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, -5.0)
    glColor3f(0.3, 0.6, 1.0)
    glutSolidSphere(1, 50, 50)
    glutSwapBuffers()

glutDisplayFunc(draw_sphere)
glutMainLoop()
#---------------------------------------------------------------------
#Şimdi seni OpenGL’in ileri düzeyi olan Shader (GLSL), VBO (Vertex Buffer Object) ve VAO (Vertex Array Object) dünyasına götürüyorum. Bu yapıların hepsi modern OpenGL’de performanslı grafikler çizmek için kullanılır. Artık sabit işlem hattı (glBegin / glEnd) yerine programlanabilir pipeline kullanıyoruz.Modern OpenGL'de Önemli Kavramlar:

#Modern OpenGL'de Önemli Kavramlar:

#Terim	Açıklama
#Shader	GPU üzerinde çalışan mini programcıklar. Genelde iki tür: Vertex ve Fragment Shader.
#GLSL	OpenGL Shader Language (C’ye benzer yapıdadır).
#VBO	Vertex verilerini GPU’ya aktarmak için kullanılır.
#VAO	Birden fazla VBO’yu tek bir yapı içinde organize eder.
#Shader Pipeline	OpenGL işlemlerinin geçtiği aşamalı yapı.


#Shader ile Üçgen (PyOpenGL + GLSL)
pip install PyOpenGL glfw
import glfw
from OpenGL.GL import *
import numpy as np

# Vertex Shader
VERTEX_SHADER = """
#version 330
layout(location = 0) in vec3 position;
void main()
{
    gl_Position = vec4(position, 1.0);
}
"""

# Fragment Shader
FRAGMENT_SHADER = """
#version 330
out vec4 FragColor;
void main()
{
    FragColor = vec4(1.0, 0.3, 0.6, 1.0); // Pembe renk
}
"""

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader

def create_shader_program():
    vertex = compile_shader(VERTEX_SHADER, GL_VERTEX_SHADER)
    fragment = compile_shader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vertex)
    glAttachShader(program, fragment)
    glLinkProgram(program)
    return program

def main():
    if not glfw.init():
        return

    window = glfw.create_window(800, 600, "Shader ile Üçgen", None, None)
    glfw.make_context_current(window)

    triangle = np.array([
        -0.5, -0.5, 0.0,
         0.5, -0.5, 0.0,
         0.0,  0.5, 0.0
    ], dtype=np.float32)

    # VAO & VBO oluştur
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)

    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, triangle.nbytes, triangle, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    shader = create_shader_program()
    glUseProgram(shader)

    while not glfw.window_should_close(window):
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glBindVertexArray(VAO)
        glDrawArrays(GL_TRIANGLES, 0, 3)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()
#---------------------------------------------

#Shader ile 3D Dönen Küp
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr

VERTEX_SHADER = """
#version 330
layout(location = 0) in vec3 position;
uniform mat4 model;
uniform mat4 projection;
void main() {
    gl_Position = projection * model * vec4(position, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
out vec4 FragColor;
void main() {
    FragColor = vec4(0.2, 0.8, 1.0, 1.0);  // Mavi ton
}
"""

vertices = [
    # Ön
    -1, -1,  1,
     1, -1,  1,
     1,  1,  1,
    -1,  1,  1,
    # Arka
    -1, -1, -1,
     1, -1, -1,
     1,  1, -1,
    -1,  1, -1,
]

indices = [
    0, 1, 2, 2, 3, 0,  # Ön
    4, 5, 6, 6, 7, 4,  # Arka
    0, 4, 7, 7, 3, 0,  # Sol
    1, 5, 6, 6, 2, 1,  # Sağ
    3, 2, 6, 6, 7, 3,  # Üst
    0, 1, 5, 5, 4, 0   # Alt
]

vertices = np.array(vertices, dtype=np.float32)
indices = np.array(indices, dtype=np.uint32)

def main():
    if not glfw.init(): return
    window = glfw.create_window(800, 600, "Dönen Küp - Shader", None, None)
    glfw.make_context_current(window)

    shader = compileProgram(
        compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    )

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)

    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    glEnable(GL_DEPTH_TEST)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(shader)

        rotation = pyrr.Matrix44.from_y_rotation(glfw.get_time())
        projection = pyrr.matrix44.create_perspective_projection_matrix(45, 800 / 600, 0.1, 100)
        projection_location = glGetUniformLocation(shader, "projection")
        model_location = glGetUniformLocation(shader, "model")

        glUniformMatrix4fv(model_location, 1, GL_FALSE, rotation)
        glUniformMatrix4fv(projection_location, 1, GL_FALSE, projection)

        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()
#-------------------------------------------------------------------
#Shader ile Işıklandırmalı Sahne (Phong Lighting)
"""Bu örnekle temel ışıklandırma (diffuse + ambient) uygulanmış bir küp göstereceğiz. Işığın geliş yönüne göre parlama efektleri göreceksin.

(İstersen bu örneği ayrıca hazırlayıp gönderebilirim. İleri ışıklandırma için vertex + fragment shader içinde normal hesapları ve lightPos, viewPos tanımları gerekir.)"""
# Mouse Kontrollü Kamera (FPS Stili)
"""FPS tarzı bir kamera için:

WASD hareket

Mouse ile bakış yönü (yaw, pitch)

Kamera matrix’ini view olarak shader’a gönderme gerekir.

Bunun için genellikle pyrr veya glm ile şu fonksiyonlar kullanılır:"""

view = pyrr.matrix44.create_look_at(camera_pos, camera_pos + camera_front, camera_up)
#----------------------------------------------------------------------------------------
#FPS (First Person Shooter)

"""FPS Kamera Kontrolü (Mouse + Klavye)
Bu sistemde:

WASD ile ilerle/geri/sağ/sol hareket

Mouse ile kamera yönü değişir (sağa-sola, yukarı-aşağı bakış)

OpenGL’e bir view matrix gönderilir

Shader üzerinden görüntü oluşturulur
pip install glfw pyrr PyOpenGL
FPS Kamera Örneği
Aşağıdaki kodda:

Küp sahnesi var (dönebilen nesneler de eklenebilir)

Kamera hareketleri mouse ve klavye ile yönetilir

OpenGL 3.3 kullanılır
"""
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
from math import sin, cos, radians

# GLSL Shader'lar
VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;
void main()
{
    FragColor = vec4(0.8, 0.2, 0.3, 1.0);
}
"""

# Kamera parametreleri
camera_pos = np.array([0.0, 0.0, 3.0], dtype=np.float32)
camera_front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

yaw = -90.0
pitch = 0.0
lastX, lastY = 400, 300
first_mouse = True

def mouse_callback(window, xpos, ypos):
    global yaw, pitch, lastX, lastY, first_mouse, camera_front

    if first_mouse:
        lastX = xpos
        lastY = ypos
        first_mouse = False

    xoffset = xpos - lastX
    yoffset = lastY - ypos
    lastX = xpos
    lastY = ypos

    sensitivity = 0.1
    xoffset *= sensitivity
    yoffset *= sensitivity

    yaw += xoffset
    pitch += yoffset
    pitch = max(-89.0, min(89.0, pitch))

    front = np.array([
        cos(radians(yaw)) * cos(radians(pitch)),
        sin(radians(pitch)),
        sin(radians(yaw)) * cos(radians(pitch))
    ], dtype=np.float32)

    global camera_front
    camera_front = front / np.linalg.norm(front)

def process_input(window):
    global camera_pos
    camera_speed = 0.05
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        camera_pos[:] += camera_speed * camera_front
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        camera_pos[:] -= camera_speed * camera_front
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        right = np.cross(camera_front, camera_up)
        camera_pos[:] -= camera_speed * right / np.linalg.norm(right)
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        right = np.cross(camera_front, camera_up)
        camera_pos[:] += camera_speed * right / np.linalg.norm(right)

def main():
    if not glfw.init():
        return

    window = glfw.create_window(800, 600, "FPS Kamera - OpenGL", None, None)
    glfw.make_context_current(window)
    glfw.set_cursor_pos_callback(window, mouse_callback)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    shader = compileProgram(
        compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    )

    cube = [
        -0.5, -0.5, -0.5,
         0.5, -0.5, -0.5,
         0.5,  0.5, -0.5,
        -0.5,  0.5, -0.5,
        -0.5, -0.5,  0.5,
         0.5, -0.5,  0.5,
         0.5,  0.5,  0.5,
        -0.5,  0.5,  0.5
    ]

    indices = [
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        0, 4, 7, 7, 3, 0,
        1, 5, 6, 6, 2, 1,
        3, 2, 6, 6, 7, 3,
        0, 1, 5, 5, 4, 0
    ]

    vertices = np.array(cube, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)

    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    glEnable(GL_DEPTH_TEST)

    while not glfw.window_should_close(window):
        process_input(window)
        glClearColor(0.1, 0.1, 0.1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(shader)

        model = pyrr.matrix44.create_identity(dtype=np.float32)
        view = pyrr.matrix44.create_look_at(camera_pos, camera_pos + camera_front, camera_up)
        projection = pyrr.matrix44.create_perspective_projection_matrix(45, 800 / 600, 0.1, 100)

        glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, model)
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, view)
        glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, projection)

        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()
#-----------------------------------------------------------------------------------------
#Şimdi seni ışıklandırmalı bir OpenGL sahnesiyle tanıştırıyorum — burada gerçek zamanlı olarak bir küp üzerine ambient, diffuse ve specular (parlama) ışık hesaplamaları yapılacak.Phong Aydınlatma Modeli Nedir?
"""Işıklandırma üç parçadan oluşur:

Ambient Light: Ortam ışığı, her zaman görülür.

Diffuse Light: Işığın geliş yönüyle yüzey normali arasındaki açıya göre belirir.

Specular Light: Gözle ışık yansıma yönünün benzerliğiyle parlama efekti."""

"""pip install glfw PyOpenGL pyrr
Aşağıdaki kod:

Küp nesnesi çizer

Bir ışık kaynağını sahneye koyar

Shader içinde Phong ışıklandırma uygular"""

import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr

vertex_src = """
#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;
out vec3 Normal;

void main()
{
    FragPos = vec3(model * vec4(a_position, 1.0));
    Normal = mat3(transpose(inverse(model))) * a_normal;

    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

fragment_src = """
#version 330 core
in vec3 FragPos;
in vec3 Normal;

out vec4 FragColor;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec3 objectColor;

void main()
{
    // Ambient
    float ambientStrength = 0.2;
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, 1.0);
}
"""

# Küp verisi (pozisyon + normal)
vertices = [
    # positions        # normals
    -0.5,-0.5,-0.5,    0.0, 0.0,-1.0,
     0.5,-0.5,-0.5,    0.0, 0.0,-1.0,
     0.5, 0.5,-0.5,    0.0, 0.0,-1.0,
     0.5, 0.5,-0.5,    0.0, 0.0,-1.0,
    -0.5, 0.5,-0.5,    0.0, 0.0,-1.0,
    -0.5,-0.5,-0.5,    0.0, 0.0,-1.0,

    -0.5,-0.5, 0.5,    0.0, 0.0, 1.0,
     0.5,-0.5, 0.5,    0.0, 0.0, 1.0,
     0.5, 0.5, 0.5,    0.0, 0.0, 1.0,
     0.5, 0.5, 0.5,    0.0, 0.0, 1.0,
    -0.5, 0.5, 0.5,    0.0, 0.0, 1.0,
    -0.5,-0.5, 0.5,    0.0, 0.0, 1.0,

    -0.5, 0.5, 0.5,   -1.0, 0.0, 0.0,
    -0.5, 0.5,-0.5,   -1.0, 0.0, 0.0,
    -0.5,-0.5,-0.5,   -1.0, 0.0, 0.0,
    -0.5,-0.5,-0.5,   -1.0, 0.0, 0.0,
    -0.5,-0.5, 0.5,   -1.0, 0.0, 0.0,
    -0.5, 0.5, 0.5,   -1.0, 0.0, 0.0,

     0.5, 0.5, 0.5,    1.0, 0.0, 0.0,
     0.5, 0.5,-0.5,    1.0, 0.0, 0.0,
     0.5,-0.5,-0.5,    1.0, 0.0, 0.0,
     0.5,-0.5,-0.5,    1.0, 0.0, 0.0,
     0.5,-0.5, 0.5,    1.0, 0.0, 0.0,
     0.5, 0.5, 0.5,    1.0, 0.0, 0.0,

    -0.5,-0.5,-0.5,    0.0,-1.0, 0.0,
     0.5,-0.5,-0.5,    0.0,-1.0, 0.0,
     0.5,-0.5, 0.5,    0.0,-1.0, 0.0,
     0.5,-0.5, 0.5,    0.0,-1.0, 0.0,
    -0.5,-0.5, 0.5,    0.0,-1.0, 0.0,
    -0.5,-0.5,-0.5,    0.0,-1.0, 0.0,

    -0.5, 0.5,-0.5,    0.0, 1.0, 0.0,
     0.5, 0.5,-0.5,    0.0, 1.0, 0.0,
     0.5, 0.5, 0.5,    0.0, 1.0, 0.0,
     0.5, 0.5, 0.5,    0.0, 1.0, 0.0,
    -0.5, 0.5, 0.5,    0.0, 1.0, 0.0,
    -0.5, 0.5,-0.5,    0.0, 1.0, 0.0,
]

vertices = np.array(vertices, dtype=np.float32)

def main():
    if not glfw.init():
        return
    window = glfw.create_window(800, 600, "Işıklandırmalı Küp", None, None)
    glfw.make_context_current(window)

    shader = compileProgram(
        compileShader(vertex_src, GL_VERTEX_SHADER),
        compileShader(fragment_src, GL_FRAGMENT_SHADER)
    )

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)

    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # Pozisyon
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    # Normal
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)

    glEnable(GL_DEPTH_TEST)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        glClearColor(0.1, 0.1, 0.1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(shader)

        # Uniformlar
        model = pyrr.matrix44.create_identity()
        view = pyrr.matrix44.create_look_at([2, 2, 3], [0, 0, 0], [0, 1, 0])
        projection = pyrr.matrix44.create_perspective_projection_matrix(45, 800/600, 0.1, 100)

        glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, model)
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, view)
        glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, projection)

        glUniform3f(glGetUniformLocation(shader, "lightPos"), 1.2, 1.0, 2.0)
        glUniform3f(glGetUniformLocation(shader, "viewPos"), 2, 2, 3)
        glUniform3f(glGetUniformLocation(shader, "lightColor"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(shader, "objectColor"), 1.0, 0.5, 0.3)

        glBindVertexArray(VAO)
        glDrawArrays(GL_TRIANGLES, 0, len(vertices) // 6)

        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()
#---------------------------------------------------------------------------------------




