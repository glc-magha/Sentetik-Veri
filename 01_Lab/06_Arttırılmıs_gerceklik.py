"""#artırılmış gerçeklik (AR) yaklaşımı )

#Sentetik Veri için 3D Model Görselleştirme

pip install open3d opencv-python numpy pillow

import open3d as o3d
import numpy as np
import cv2
from PIL import Image
import os

# Klasör ayarları
output_folder = "sentetik_veri"
os.makedirs(output_folder, exist_ok=True)

# 3D model yükle (örneğin: çaydanlık obj)
mesh = o3d.io.read_triangle_mesh("teapot.obj")
mesh.compute_vertex_normals()

# Görüntülerin üretileceği farklı açılar
angles = np.linspace(0, 2 * np.pi, 24)  # 24 farklı açı

# Görüntü oluşturucu
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)
vis.add_geometry(mesh)

# Kamera kontrolü
ctr = vis.get_view_control()
param = ctr.convert_to_pinhole_camera_parameters()

# Arka planlar (isteğe bağlı PNG veya başka arka planlar ekleyebilirsin)
backgrounds = ["bg1.jpg", "bg2.jpg", "bg3.jpg"]  # Bu dosyalarklasöre eklenmeli

for i, angle in enumerate(angles):
    R = mesh.get_rotation_matrix_from_xyz((0, angle, 0))
    mesh.rotate(R, center=mesh.get_center())

    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()

    # Ekran görüntüsü al
    img = vis.capture_screen_float_buffer(do_render=True)
    img_np = (255 * np.asarray(img)).astype(np.uint8)

    # Arka plan resmi ekleme
    bg_path = np.random.choice(backgrounds)
    background = cv2.imread(bg_path)
    background = cv2.resize(background, (img_np.shape[1], img_np.shape[0]))

    # Alpha blending ile sahte AR etkisi (varsayılan opaklık)
    blended = cv2.addWeighted(img_np, 0.7, background, 0.3, 0)

    # Görüntüyü kaydet
    out_path = os.path.join(output_folder, f"teapot_{i}.jpg")
    cv2.imwrite(out_path, blended)
    print(f"{out_path} kaydedildi.")

vis.destroy_window()
"""