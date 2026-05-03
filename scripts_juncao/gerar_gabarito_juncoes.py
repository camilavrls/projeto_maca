"""
Gera um gabarito de referência com as posições e ângulos
das peças em uma montagem CORRETA.
"""

import numpy as np
import cv2
from ultralytics import YOLO

SEG_MODEL_PATH = "../best-seg.pt"
GABARITO_NPZ = "gabarito_seg.npz"
OUT_DIR = "avaliacoes_seg"

seg_model = YOLO(SEG_MODEL_PATH)

def processar_imagem(img_path: str) -> dict:
    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    resultado = seg_model(img_rgb)[0]

    boxes = resultado.boxes.xyxy.cpu().numpy()
    masks = resultado.masks.data.cpu().numpy()
    classes = resultado.boxes.cls.cpu().numpy()
    confiancas = resultado.boxes.conf.cpu().numpy()

    nome_classes = []
    for c in classes:
        nome_classes.append(seg_model.names[int (c)])

    return {
        "imagem": img_rgb,
        "boxes": boxes,
        "masks": masks,
        "classes": nome_classes,
        "confiancas": confiancas
    }

def calcular_centroide_normalizado(masks, boxes) -> tuple:
    x_min = np.min(boxes[:, 0])
    y_min = np.min(boxes[:, 1])
    x_max = np.max(boxes[:, 2])
    y_max = np.max(boxes[:, 3])
    
    largura_total = x_max - x_min
    altura_total = y_max - y_min
    
    centroides_abs = []
    centroides_normalizados = []
    
    for i, mascara in enumerate(masks):
        y_pixels, x_pixels = np.where(mascara == 1)
        
        cx_abs = np.mean(x_pixels)
        cy_abs = np.mean(y_pixels)

        centroides_abs.append([cx_abs, cy_abs])
        
        cx_norm = (cx_abs - x_min) / largura_total
        cy_norm = (cy_abs - y_min) / altura_total
        
        centroides_normalizados.append([cx_norm, cy_norm])

    return np.array(centroides_abs), np.array(centroides_normalizados)

def calcular_angulo_orientacao(masks, centroides_abs) -> np.ndarray:
    angulos = []
    
    for i, mascara in enumerate(masks):
        
        y_pixels, x_pixels = np.where(mascara == 1)
        
        cx_abs = centroides_abs[i, 0]
        cy_abs = centroides_abs[i, 1]
        
        distancias = np.sqrt((x_pixels - cx_abs)**2 + (y_pixels - cy_abs)**2)
        idx_mais_distante = np.argmax(distancias)
        
        x_dist = x_pixels[idx_mais_distante]
        y_dist = y_pixels[idx_mais_distante]
        
        dx = x_dist - cx_abs
        dy = y_dist - cy_abs
        
        angulo_rad = np.arctan2(dy, dx)
        angulo_deg = np.degrees(angulo_rad)
        
        if angulo_deg < 0:
            angulo_deg += 360
        
        angulos.append(angulo_deg)
    
    return np.array(angulos)

if __name__ == "__main__":
    resultado = processar_imagem("correta.jpg")

    print(resultado["classes"])
    print(resultado["boxes"])
    print(resultado["confiancas"])
    print(resultado["masks"])
