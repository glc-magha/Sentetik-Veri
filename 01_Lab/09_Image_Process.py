""""""
"""TokenCut, görüntü ve videolardaki nesneleri segmentlemek için kendiliğinden denetimli (self-supervised) transformer özelliklerini ve Normalized Cut algoritmasını kullanan bir yöntemdir. Bu yaklaşımda, görüntü yamaları (patch'ler) bir grafın düğümleri olarak ele alınır ve düğümler arasındaki kenarlar, transformer tarafından öğrenilen özelliklere dayalı benzerlik skorlarıyla ağırlıklandırılır. Daha sonra, bu graf üzerinde Normalized Cut algoritması uygulanarak, öne çıkan nesneler tespit edilir ve segmentlenir. ​

Bu kod, bir görüntüyü yükleyip ön işler, önceden eğitilmiş bir Vision Transformer (ViT) modeli kullanarak özellikleri çıkarır, bu özelliklere dayalı bir benzerlik matrisi oluşturur ve ardından Normalized Cut algoritmasıyla spektral kümeleme yaparak nesne segmentasyonu gerçekleştirir. Bu, TokenCut yönteminin temel prensiplerini basit bir örnekle göstermektedir

​TokenCut, görüntü ve videolardaki nesneleri segmentlemek için kendiliğinden denetimli (self-supervised) bir transformer modeli ve Normalize Kesim (Normalized Cut) algoritmasını kullanan bir yöntemdir. Bu yöntemde, görüntü veya videoyu oluşturan yama (patch) lar, öğrenilmiş özelliklere dayalı benzerlik skorlarıyla tam bağlantılı bir grafik olarak modellenir. Nesne tespiti ve segmentasyonu, bu grafikte bir kesim problemi olarak ele alınır ve klasik Normalize Kesim algoritmasıyla çözülür."""
"""import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models import vit_b_16
from sklearn.cluster import SpectralClustering
import numpy as np
#Görüntü segmentasyonu
# Görüntüyü yükleyin ve ön işleyin
image_path = 'path_to_your_image.jpg'
image = Image.open(image_path).convert('RGB')
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
input_tensor = transform(image).unsqueeze(0)

# Önceden eğitilmiş bir Vision Transformer modelini yükleyin
model = vit_b_16(pretrained=True)
model.eval()

# Görüntü özelliklerini çıkarın
with torch.no_grad():
    features = model(input_tensor)

# Özellikleri yeniden şekillendirin
features = features.squeeze(0).permute(1, 2, 0).view(-1, features.shape[1])

# Özellikler arasındaki benzerlik matrisini hesaplayın
affinity_matrix = torch.mm(features, features.t()).numpy()

# Normalized Cut kullanarak spektral kümeleme yapın
sc = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='discretize')
labels = sc.fit_predict(affinity_matrix)

# Sonuçları görselleştirin
segmentation = labels.reshape(14, 14)  # 224x224 görüntü için 14x14 patch'ler
import matplotlib.pyplot as plt
plt.imshow(segmentation)
plt.show()
"""

import torch
import timm
from PIL import Image
import torchvision.transforms as T
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import numpy as np

# Görüntüyü yükle ve dönüştür
image_path = 'path_to_your_image.jpg'
image = Image.open(image_path).convert('RGB')
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
input_tensor = transform(image).unsqueeze(0)

# ViT modelini yükle
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

# Özellikleri çıkar (class token'ı çıkar, sadece patch'ler)
with torch.no_grad():
    outputs = model.forward_features(input_tensor)  # (B, N+1, C)
    patch_features = outputs[:, 1:, :]  # class token hariç (1:), shape: (1, 196, C)

# 196 patch (14x14) → her patch için C boyutlu vektör
patch_features = patch_features.squeeze(0).cpu().numpy()  # shape: (196, C)

# Benzerlik matrisi
affinity_matrix = np.matmul(patch_features, patch_features.T)

# Spektral Kümeleme
sc = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='discretize')
labels = sc.fit_predict(affinity_matrix)

# 14x14 segmente et
segmentation = labels.reshape(14, 14)

# Görüntüle
plt.imshow(segmentation, cmap='viridis')
plt.axis('off')
plt.show()
"""

#-----------------------------------------------------------------
#görüntü okuma
"""import cv2

# Görüntüyü yükle
image = cv2.imread('image.jpg')

# Görüntüyü göster
cv2.imshow('Görüntü', image)

# Bir tuşa basılana kadar bekle
cv2.waitKey(0)

# Pencereleri kapat
cv2.destroyAllWindows()"""

import cv2

# Görüntüyü yükle
image = cv2.imread('image.jpg')

# Eğer görüntü yüklenememişse hata mesajı ver
if image is None:
    print("Görüntü yüklenemedi! Lütfen dosya yolunun doğru olduğundan emin olun.")
else:
    # Görüntüyü göster
    cv2.imshow('Görüntü', image)

    # Bir tuşa basılana kadar bekle
    cv2.waitKey(0)

    # Pencereleri kapat
    cv2.destroyAllWindows()

#-----------------------------------------------------#
    #Grayscale gri tonlama
import cv2

# Görüntüyü yükle
image = cv2.imread('image.jpg')

# Görüntüyü gri tonlamaya çevir
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gri tonlamalı görüntüyü göster
cv2.imshow('Gri Tonlama', gray_image)

# Bir tuşa basılana kadar bekle
cv2.waitKey(0)

# Pencereleri kapat
cv2.destroyAllWindows()
#---------------------------------------------------------------------"""