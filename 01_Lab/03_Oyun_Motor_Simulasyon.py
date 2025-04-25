"""#Unity veya bir oyun motorunda çalışan simülasyon, farklı sahneler oluşturuyor. Python ile bu sahnelerden:
#Kamera görüntüsü alacağız
#Nesne koordinatlarını ve etiketlerini kaydedeceğiz (bounding box)
#Sentetik veri olarak JSON + Görsel çıktısı üreteceğiz


#Unity veya simülasyon sahnesi bu verileri yolluyor
{
  "image": "<base64_encoded_image>",
  "objects": [
    {"class": "car", "bbox": [100, 150, 200, 300]},
    {"class": "person", "bbox": [400, 120, 460, 300]}
  ]
}
#Python Kod – Veriyi al, çözümle, kaydet

import socket
import base64
import json
import cv2
import numpy as np
import os

# Klasörler
output_folder = "sentetik_data"
os.makedirs(output_folder, exist_ok=True)

# Socket Ayarları
HOST = "localhost"
PORT = 5050

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(5)

print(f"Simülasyon bağlantısı bekleniyor: {HOST}:{PORT}")

counter = 0

while True:
    conn, addr = server.accept()
    print(f"Bağlandı: {addr}")

    data = conn.recv(10 ** 6)  # 1MB veri al
    conn.close()

    try:
        decoded = json.loads(data.decode('utf-8'))

        # Görseli çöz
        img_data = base64.b64decode(decoded["image"])
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Görüntü kaydet
        image_path = os.path.join(output_folder, f"img_{counter}.jpg")
        cv2.imwrite(image_path, image)

        # Etiket dosyası (JSON)
        labels_path = os.path.join(output_folder, f"img_{counter}.json")
        with open(labels_path, 'w') as f:
            json.dump(decoded["objects"], f, indent=4)

        print(f"{counter} -> Kayıt tamamlandı: {image_path}")
        counter += 1

    except Exception as e:
        print("Hata:", e)
##Üretilen Veri Yapısı Örneği
#sentetik_data/
├── img_0.jpg
├── img_0.json  ← {"class": "car", "bbox": [...]}
├── img_1.jpg
├── img_1.json
...


"""