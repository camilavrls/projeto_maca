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

if __name__ == "__main__":
    resultado = processar_imagem("correta.jpg")

    print(resultado["classes"])
    print(resultado["boxes"])
    print(resultado["confiancas"])
    print(resultado["masks"])
