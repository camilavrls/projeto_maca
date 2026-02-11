import numpy as np
import cv2
from ultralytics import YOLO

SEG_MODEL_PATH = "best-seg.pt"       
IMG_REF_PATH   = "correta.jpg"       # imagem referência correta para o gabarito
OUT_NPZ_PATH   = "gabarito_seg.npz"  # arquivo de saída com gabarito


seg_model = YOLO(SEG_MODEL_PATH)


def calcular_angulo_mascara(mask_bin: np.ndarray) -> float:
    M = cv2.moments(mask_bin)
    mu11 = M["mu11"]
    mu20 = M["mu20"]
    mu02 = M["mu02"]

    if (mu20 - mu02) == 0 and mu11 == 0:
        angle_deg = 0.0
    else:
        angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)  # radianos
        angle_deg = np.degrees(angle)

    angle_deg = (angle_deg + 180.0) % 180.0

    # normaliza para [0,1]
    angle_norm = angle_deg / 180.0
    return angle_norm


def gerar_gabarito_seg(img_path: str):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Não consegui carregar a imagem: {img_path}")

    results = seg_model(img)[0]

    if len(results.boxes) == 0:
        raise RuntimeError("Nenhuma peça segmentada na imagem de referência.")

    boxes = results.boxes.xyxy.cpu().numpy()  # (N, 4)
    cls = results.boxes.cls.cpu().numpy()     # (N,)
    conf = results.boxes.conf.cpu().numpy()   # (N,)

    if results.masks is None:
        raise RuntimeError("O modelo de segmentação não retornou máscaras.")

    masks = results.masks.data.cpu().numpy()  # (N, H, W)

    # bounding box global da maçã (união das peças)
    x1_global = boxes[:, 0].min()
    y1_global = boxes[:, 1].min()
    x2_global = boxes[:, 2].max()
    y2_global = boxes[:, 3].max()

    width_global  = (x2_global - x1_global) + 1e-6
    height_global = (y2_global - y1_global) + 1e-6

    classes_unicas = np.unique(cls).astype(int)
    centroids_ref_norm = []
    classes_ref = []
    angles_ref_norm = []

    for c in classes_unicas:
        idxs = np.where(cls == c)[0]
        # escolhe a detecção com maior confiança
        best_idx = idxs[np.argmax(conf[idxs])]
        x1, y1, x2, y2 = boxes[best_idx]

        # centróide em coords globais
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # centróide normalizado dentro da maçã
        cx_n = (cx - x1_global) / width_global
        cy_n = (cy - y1_global) / height_global

        # matriz de probabilidades de ser fundo 0 ou peça 1
        mask = masks[best_idx]           
        mask_bin = (mask > 0.5).astype("uint8")
        angle_n = calcular_angulo_mascara(mask_bin)

        centroids_ref_norm.append([cx_n, cy_n])
        classes_ref.append(c)
        angles_ref_norm.append(angle_n)

    centroids_ref_norm = np.array(centroids_ref_norm)  # (K, 2)
    classes_ref = np.array(classes_ref, dtype=int)     # (K,)
    angles_ref_norm = np.array(angles_ref_norm, dtype=float)  # (K,)

    return classes_ref, centroids_ref_norm, angles_ref_norm


if __name__ == "__main__":
    print(f"Gerando gabarito de segmentação a partir de: {IMG_REF_PATH}")
    classes_ref, centroids_ref_norm, angles_ref_norm = gerar_gabarito_seg(IMG_REF_PATH)

    np.savez(
        OUT_NPZ_PATH,
        classes=classes_ref,
        centroids=centroids_ref_norm,
        angles=angles_ref_norm,
    )

    print(f"Gabarito de segmentação salvo em: {OUT_NPZ_PATH}")
    print("Classes de referência:", classes_ref)
    print("Centróides normalizados de referência:\n", centroids_ref_norm)
    print("Ângulos normalizados de referência (0–1, equivale a 0–180°):\n", angles_ref_norm)