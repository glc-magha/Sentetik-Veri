"""#Stereo Kameradan 3D Derinlik Haritası

pip install opencv-python opencv-contrib-python
#Stereo Görüntülerle Derinlik Haritası Hesaplama

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Stereo görüntüleri yükle (önceden hizalanmış olmalı)
imgL = cv2.imread('left.jpg', 0)  # sol kamera
imgR = cv2.imread('right.jpg', 0)  # sağ kamera

# StereoSGBM parametreleri
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,  # 16'nın katı olmalı
    blockSize=5,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    disp12MaxDiff=1,
    P1=8 * 3 * 5 ** 2,
    P2=32 * 3 * 5 ** 2
)

# Eşleşme (disparity) haritası
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

# Görselleştir
plt.imshow(disparity, cmap='plasma')
plt.colorbar()
plt.title("Derinlik Haritası")
plt.show()



#......................................................
#İleri Seviye: Derinlikten 3D Nokta Bulutu Üretme
# Kamera parametreleri (örnek değerler)
focal_length = 800  # px
baseline = 0.06  # metre

# Derinlik hesabı
depth_map = (focal_length * baseline) / (disparity + 1e-5)

# Derinliği görselleştir
plt.imshow(depth_map, cmap='inferno')
plt.title("Mesafe Haritası (metre)")
plt.colorbar()
plt.show()
#Intel RealSense ile 3D Görü
import pyrealsense2 as rs

pipeline = rs.pipeline()
pipeline.start()

while True:
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()

    if not depth or not color:
        continue

    depth_image = np.asanyarray(depth.get_data())
    color_image = np.asanyarray(color.get_data())

    # Derinlik skalası
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    cv2.imshow('Depth', depth_colormap)
    if cv2.waitKey(1) == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()


"""