import numpy as np
import cv2
import os
from ultralytics import YOLO

SEG_MODEL_PATH  = "best-seg.pt"
GABARITO_NPZ    = "gabarito_seg.npz"

TOL_POS       = 0.15   # tolerância de posição normalizada antes de considerar erro máximo
TOL_ANG       = 0.15   # tolerância de diferença angular normalizada (~27°) antes de erro máximo
PESO_POS      = 0.5    # peso da posição no erro da peça
PESO_ANG      = 0.5    # peso da rotação no erro da peça

PENALIZACAO_PECAS = 0.3    # quanto a falta de peças pesa na nota final
OUT_DIR           = "avaliacoes_seg"

seg_model = YOLO(SEG_MODEL_PATH)

data = np.load(GABARITO_NPZ)
classes_ref        = data["classes"]        # (K,)
centroids_ref_norm = data["centroids"]     # (K, 2)
angles_ref_norm    = data["angles"]        # (K,)

PECAS_ESPERADAS = len(classes_ref)

print("Gabarito carregado:")
print("Classes de referência:", classes_ref)
print("Centróides normalizados de referência:\n", centroids_ref_norm)
print("Ângulos normalizados de referência (0–1 → 0–180°):\n", angles_ref_norm)


def desenhar_ponto(img, x, y, cor, radius=7):
    cv2.circle(img, (int(x), int(y)), radius, cor, -1)


def desenhar_texto(img, texto, pos=(20, 40)):
    cv2.putText(img, texto, pos, cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 3)
    cv2.putText(img, texto, pos, cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 1)


def calcular_angulo_mascara(mask_bin: np.ndarray) -> float:
    M = cv2.moments(mask_bin)
    mu11 = M["mu11"]
    mu20 = M["mu20"]
    mu02 = M["mu02"]

    if (mu20 - mu02) == 0 and mu11 == 0:
        angle_deg = 0.0
    else:
        angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
        angle_deg = np.degrees(angle)

    angle_deg = (angle_deg + 180.0) % 180.0
    angle_norm = angle_deg / 180.0
    return angle_norm


def avaliar_montagem_seg(img_path: str, salvar=True) -> float:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Não consegui carregar a imagem: {img_path}")

    results = seg_model(img)[0]
    img_plot = results.plot().copy()

    if len(results.boxes) == 0:
        print(f"[{img_path}] Nenhuma peça segmentada.")
        nota = 0.0
        if salvar:
            os.makedirs(OUT_DIR, exist_ok=True)
            out_path = os.path.join(
                OUT_DIR, os.path.basename(img_path).replace(".jpg", "_seg_avaliada.jpg")
            )
            desenhar_texto(img_plot, f"Nota: {nota}%")
            cv2.imwrite(out_path, img_plot)
        return nota

    if results.masks is None:
        print(f"[{img_path}] Modelo não retornou máscaras.")
        return 0.0

    boxes = results.boxes.xyxy.cpu().numpy()  # (N,4)
    cls   = results.boxes.cls.cpu().numpy()   # (N,)
    conf  = results.boxes.conf.cpu().numpy()  # (N,)
    masks = results.masks.data.cpu().numpy()  # (N,H,W)

    # bounding box global da maçã
    x1_global = boxes[:, 0].min()
    y1_global = boxes[:, 1].min()
    x2_global = boxes[:, 2].max()
    y2_global = boxes[:, 3].max()

    width_global  = (x2_global - x1_global) + 1e-6
    height_global = (y2_global - y1_global) + 1e-6

    erros_pecas = []   # erro combinado (posição + rotação) por peça [0,1]
    presentes = 0

    centroids_detectados_global = []
    centroids_ref_global = []

    for idx_ref, c_ref in enumerate(classes_ref):
        cx_ref_n, cy_ref_n = centroids_ref_norm[idx_ref]
        ang_ref_n = angles_ref_norm[idx_ref]

        idxs = np.where(cls == c_ref)[0]

        if len(idxs) == 0:
            # peça não apareceu: erro máximo
            erros_pecas.append(1.0)
            print(f"[{img_path}] Classe {c_ref}: peça ausente → erro=1.0")
            continue

        presentes += 1

        # detecção mais confiável dessa classe
        best_idx = idxs[np.argmax(conf[idxs])]
        x1, y1, x2, y2 = boxes[best_idx]
        mask = masks[best_idx]
        mask_bin = (mask > 0.5).astype("uint8")

        # centróide detectado
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # normalizado pela maçã
        cx_n = (cx - x1_global) / width_global
        cy_n = (cy - y1_global) / height_global

        # ângulo detectado
        ang_det_n = calcular_angulo_mascara(mask_bin)

        # erro de posição (normalizado pela tolerância e truncado em 1)
        d_pos = np.linalg.norm([cx_n - cx_ref_n, cy_n - cy_ref_n])
        d_pos_norm = min(d_pos / TOL_POS, 1.0)

        # erro de rotação: diferença angular circular
        diff_ang = abs(ang_det_n - ang_ref_n)
        diff_ang = min(diff_ang, 1.0 - diff_ang)   # por simetria, máx = 0.5
        d_ang_norm = min(diff_ang / TOL_ANG, 1.0)

        # erro combinado da peça
        d_comb = PESO_POS * d_pos_norm + PESO_ANG * d_ang_norm
        erros_pecas.append(d_comb)

        # guardar para visualização
        centroids_detectados_global.append((cx, cy))
        cx_ref_global = cx_ref_n * width_global  + x1_global
        cy_ref_global = cy_ref_n * height_global + y1_global
        centroids_ref_global.append((cx_ref_global, cy_ref_global))

        print(
            f"[{img_path}] Classe {c_ref}: "
            f"d_pos={d_pos:.3f} (norm={d_pos_norm:.3f}), "
            f"d_ang={diff_ang:.3f} (norm={d_ang_norm:.3f}), "
            f"d_comb={d_comb:.3f}, "
            f"ang_det={ang_det_n*180:.1f}°, ang_ref={ang_ref_n*180:.1f}°"
        )

    if len(erros_pecas) == 0:
        nota = 0.0
    else:
        erro_medio = np.mean(erros_pecas)  # 0 = perfeito, 1 = muito ruim
        score_pos_rot = max(0.0, 1.0 - erro_medio)

        # fator por peças presentes
        if PECAS_ESPERADAS > 0:
            fator_pecas = min(1.0, presentes / PECAS_ESPERADAS)
        else:
            fator_pecas = 1.0

        nota = (1 - PENALIZACAO_PECAS) * score_pos_rot + PENALIZACAO_PECAS * fator_pecas
        nota = round(nota * 100, 2)

    # desenhar pontos e nota
    for (cx, cy) in centroids_detectados_global:
        desenhar_ponto(img_plot, cx, cy, (0, 0, 255))   # vermelho = detectado

    for (cxr, cyr) in centroids_ref_global:
        desenhar_ponto(img_plot, cxr, cyr, (255, 0, 0)) # azul = referência

    desenhar_texto(img_plot, f"Nota: {nota}%")

    if salvar:
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(
            OUT_DIR, os.path.basename(img_path).replace(".jpg", "_seg_avaliada.jpg")
        )
        cv2.imwrite(out_path, img_plot)
        print(f"[OK] Imagem avaliada salva em: {out_path}")

    print(f"[{img_path}] erros_pecas = {erros_pecas}, presentes = {presentes}, nota = {nota}%\n")

    return nota


if __name__ == "__main__":
    img_correta    = "correta.jpg"
    img_incorreta  = "incorreta.jpg"
    img_incorreta2 = "incorreta2.jpg"  # opcional, se você tiver

    nota_c  = avaliar_montagem_seg(img_correta)
    nota_i  = avaliar_montagem_seg(img_incorreta)
    try:
        nota_i2 = avaliar_montagem_seg(img_incorreta2)
    except FileNotFoundError:
        nota_i2 = None

    print("\n===== RESULTADOS FINAIS (SEGMENTAÇÃO + POSIÇÃO + ÂNGULO) =====")
    print(f"Correta:   {nota_c}")
    print(f"Incorreta: {nota_i}")
    if nota_i2 is not None:
        print(f"Incorreta2: {nota_i2}")