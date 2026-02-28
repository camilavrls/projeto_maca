import numpy as np
import cv2
import os
from ultralytics import YOLO

# =========================================================
# Avaliação híbrida: Segmentação (posição/proximidade/fechamento)
# + Keypoints (rotação, com fallback para máscara)
# =========================================================

# =========================
# Parâmetros (ajuste aqui)
# =========================
SEG_MODEL_PATH = "best-seg.pt"
KP_MODEL_PATH  = "best-kp.pt"          # <<<< SEU MODELO DE KEYPOINTS (pose)
GABARITO_NPZ   = "gabarito_seg.npz"
OUT_DIR        = "avaliacoes_seg_kp"

# Tolerâncias
TOL_POS  = 0.05     # <<< NOVO: tolerância p/ posição (coords normalizadas 0–1) (recomendado)
TOL_ROT_DEG = 10.0  # rotação: até 10° não penaliza
TOL_PROX = 0.08     # proximidade: tolerância em coordenadas normalizadas (0–1)
TOL_FECH = 0.06     # fechamento: tolerância em coordenadas normalizadas (0–1)

# Pesos (devem somar 1)
W_POS  = 0.35
W_PROX = 0.25
W_ROT  = 0.25
W_FECH = 0.15

# Predição keypoints
KP_CONF = 0.20
KP_IOU  = 0.50
KP_MAX_DET = 5

# =========================

seg_model = YOLO(SEG_MODEL_PATH)
kp_model  = YOLO(KP_MODEL_PATH)


# =========================
# Utilidades
# =========================
def _assert_pesos():
    s = W_POS + W_PROX + W_ROT + W_FECH
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Pesos devem somar 1.0 (atual={s}). Ajuste W_POS/W_PROX/W_ROT/W_FECH.")


def desenhar_ponto(img, x, y, cor, radius=7):
    cv2.circle(img, (int(x), int(y)), radius, cor, -1)


def desenhar_texto(img, texto, pos=(20, 40)):
    cv2.putText(img, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    cv2.putText(img, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)


# =========================
# Ângulos
# =========================
def _norm_angle_360(angle_deg: float) -> float:
    return float((angle_deg + 360.0) % 360.0)


def calcular_angulo_mascara(mask_bin: np.ndarray) -> float:
    """Retorna ângulo em graus [0, 360). (PCA via momentos)"""
    M = cv2.moments(mask_bin)
    mu11 = M.get("mu11", 0.0)
    mu20 = M.get("mu20", 0.0)
    mu02 = M.get("mu02", 0.0)

    if (mu20 - mu02) == 0 and mu11 == 0:
        return 0.0

    angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    angle_deg = np.degrees(angle)
    return _norm_angle_360(angle_deg)


def calcular_angulo_keypoints(kps_xy: np.ndarray) -> float:
    """Ângulo em graus [0,360) a partir de keypoints.
    Estratégia simples: usa os 2 primeiros keypoints válidos e cria um vetor.
    """
    if kps_xy is None:
        return 0.0

    pts = []
    for p in np.array(kps_xy).reshape(-1, 2):
        x, y = float(p[0]), float(p[1])
        if np.isfinite(x) and np.isfinite(y) and not (x == 0.0 and y == 0.0):
            pts.append((x, y))
        if len(pts) >= 2:
            break

    if len(pts) < 2:
        return 0.0

    (x1, y1), (x2, y2) = pts[:2]
    ang = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
    return _norm_angle_360(ang)


def diff_ang_real(a: float, b: float) -> float:
    """Diferença angular REAL em graus, penaliza 180° corretamente (0..180)."""
    d = abs(float(a) - float(b)) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return float(d)


# =========================
# Gabarito
# =========================
def carregar_gabarito(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    classes_ref = data["classes"].astype(int)          # (K,)
    centroids_ref_norm = data["centroids"].astype(float)  # (K,2)

    angles_ref = data["angles"].astype(float) if "angles" in data.files else np.zeros(len(classes_ref), dtype=float)

    # Heurística: se ângulo estiver em [0,1], converte para graus. Se já estiver em graus, mantém.
    if np.nanmax(angles_ref) <= 1.0 + 1e-6:
        angles_ref_deg = angles_ref * 180.0
    else:
        angles_ref_deg = angles_ref

    # OBS: Se você gerar gabarito com máscara (momento), ele pode estar "mod 180".
    # Aqui mantemos como veio e comparamos por diff_ang_real (0..180).
    return classes_ref, centroids_ref_norm, angles_ref_deg


# =========================
# Arestas (vizinhança) do gabarito
# =========================
def _build_edges_knn(centroids_ref_norm: np.ndarray, k: int = 3):
    K = len(centroids_ref_norm)
    edges = set()
    for i in range(K):
        dists = []
        for j in range(K):
            if i == j:
                continue
            d = float(np.linalg.norm(centroids_ref_norm[i] - centroids_ref_norm[j]))
            dists.append((d, j))
        dists.sort(key=lambda x: x[0])
        for _, j in dists[:k]:
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))

    edges = sorted(list(edges))

    max_ref = 0.0
    ref_d = {}
    for (i, j) in edges:
        d = float(np.linalg.norm(centroids_ref_norm[i] - centroids_ref_norm[j]))
        ref_d[(i, j)] = d
        max_ref = max(max_ref, d)
    max_ref = max(max_ref, 1e-6)
    return edges, ref_d, max_ref


# =========================
# Extração: segmentação + keypoints (no crop)
# =========================
def _extrair_deteccoes(img: np.ndarray, draw_kp: bool = True):
    """
    Roda segmentação e retorna:
      - img_plot: plot do resultado + keypoints desenhados (opcional)
      - det_por_classe: {class_id: {'cx_n','cy_n','cx','cy','ang_deg','conf','ang_src'}}
      - bbox_global: (x1g,y1g,wg,hg) para normalização
    """
    seg_res = seg_model(img)[0]
    img_plot = seg_res.plot().copy()

    if len(seg_res.boxes) == 0 or seg_res.masks is None:
        return img_plot, {}, None

    boxes = seg_res.boxes.xyxy.cpu().numpy()  # (N,4)
    cls   = seg_res.boxes.cls.cpu().numpy()   # (N,)
    conf  = seg_res.boxes.conf.cpu().numpy()  # (N,)
    masks = seg_res.masks.data.cpu().numpy()  # (N,H,W)

    x1g = float(boxes[:, 0].min())
    y1g = float(boxes[:, 1].min())
    x2g = float(boxes[:, 2].max())
    y2g = float(boxes[:, 3].max())
    wg = float((x2g - x1g) + 1e-6)
    hg = float((y2g - y1g) + 1e-6)

    det_por_classe = {}

    # mantém 1 detecção por classe (maior conf)
    for i in range(len(cls)):
        c = int(cls[i])
        if (c in det_por_classe) and (conf[i] <= det_por_classe[c]["conf"]):
            continue

        x1, y1, x2, y2 = boxes[i]
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)

        cx_n = float((cx - x1g) / wg)
        cy_n = float((cy - y1g) / hg)

        # --- keypoints no crop ---
        ang_deg = 0.0
        ang_src = "mask"
        kps_global = None

        xi1, yi1, xi2, yi2 = int(x1), int(y1), int(x2), int(y2)
        xi1 = max(0, xi1); yi1 = max(0, yi1)
        xi2 = min(img.shape[1], xi2); yi2 = min(img.shape[0], yi2)

        crop = img[yi1:yi2, xi1:xi2]

        if crop.size > 0:
            kp_res = kp_model.predict(
                crop,
                conf=KP_CONF,
                iou=KP_IOU,
                max_det=KP_MAX_DET,
                verbose=False
            )[0]

            if kp_res.boxes is not None and len(kp_res.boxes) > 0 and kp_res.keypoints is not None:
                idx_best = int(np.argmax(kp_res.boxes.conf.cpu().numpy()))
                try:
                    kps = kp_res.keypoints.xy.cpu().numpy()[idx_best]  # (K,2) no crop
                    ang_deg = calcular_angulo_keypoints(kps)
                    ang_src = "kp"

                    # converter para coords globais
                    kps_global = kps.copy()
                    kps_global[:, 0] += float(xi1)
                    kps_global[:, 1] += float(yi1)

                    if draw_kp:
                        # desenha pontos
                        for (px, py) in kps_global:
                            cv2.circle(img_plot, (int(px), int(py)), 6, (0, 255, 255), -1)  # amarelo
                        # desenha linha entre 2 primeiros pontos válidos (se existirem)
                        if len(kps_global) >= 2:
                            p1 = (int(kps_global[0, 0]), int(kps_global[0, 1]))
                            p2 = (int(kps_global[1, 0]), int(kps_global[1, 1]))
                            cv2.line(img_plot, p1, p2, (0, 255, 255), 2)
                except Exception:
                    ang_deg = 0.0
                    ang_src = "mask"

        # fallback: máscara
        if ang_deg == 0.0:
            mask = masks[i]
            mask_bin = (mask > 0.5).astype("uint8")
            ang_deg = calcular_angulo_mascara(mask_bin)

        # etiqueta debug
        if draw_kp:
            cv2.putText(
                img_plot,
                f"ang={ang_deg:.1f}({ang_src})",
                (int(x1), max(0, int(y1) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255) if ang_src == "kp" else (255, 255, 255),
                2
            )

        det_por_classe[c] = {
            "cx": cx, "cy": cy,
            "cx_n": cx_n, "cy_n": cy_n,
            "ang_deg": float(ang_deg),
            "conf": float(conf[i]),
            "ang_src": ang_src,
            "kps_global": kps_global
        }

    bbox = (x1g, y1g, wg, hg)
    return img_plot, det_por_classe, bbox


# =========================
# Avaliação
# =========================
def avaliar_montagem(img_path: str, salvar: bool = True, draw_kp: bool = True) -> float:
    _assert_pesos()

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Não consegui carregar a imagem: {img_path}")

    classes_ref, centroids_ref_norm, angles_ref_deg = carregar_gabarito(GABARITO_NPZ)
    K = len(classes_ref)

    img_plot, det_por_classe, bbox = _extrair_deteccoes(img, draw_kp=draw_kp)

    if bbox is None:
        print(f"[{img_path}] Nenhuma peça segmentada.")
        nota = 0.0
        if salvar:
            os.makedirs(OUT_DIR, exist_ok=True)
            out_path = os.path.join(OUT_DIR, os.path.basename(img_path).replace(".jpg", "_avaliada.jpg"))
            desenhar_texto(img_plot, f"Nota: {nota}%")
            cv2.imwrite(out_path, img_plot)
        return nota

    x1g, y1g, wg, hg = bbox

    # Arrays por idx_ref (ordem do gabarito)
    det_centroids_norm_por_ref = [None] * K
    det_centroids_global_por_ref = [None] * K
    det_angles_por_ref = [None] * K
    det_angsrc_por_ref = [None] * K

    presentes = 0

    # ---------------------------------------------------------
    # 1) POSIÇÃO (corrigida): distância da classe para o slot da própria classe
    #    -> NÃO usa mais "slot_pred", que causava inversões.
    #    -> Com tolerância TOL_POS (recomendado).
    # ---------------------------------------------------------
    pos_penalties = []
    pos_ok_hard = 0  # só para debug

    for idx_ref, c_ref in enumerate(classes_ref):
        c_ref_int = int(c_ref)

        ref_xy = centroids_ref_norm[idx_ref]

        if c_ref_int not in det_por_classe:
            pos_penalties.append(1.0)  # faltou peça
            continue

        presentes += 1
        d = det_por_classe[c_ref_int]
        det_xy = np.array([d["cx_n"], d["cy_n"]], dtype=float)

        det_centroids_norm_por_ref[idx_ref] = det_xy
        det_centroids_global_por_ref[idx_ref] = (d["cx"], d["cy"])
        det_angles_por_ref[idx_ref] = float(d["ang_deg"])
        det_angsrc_por_ref[idx_ref] = d["ang_src"]

        dist = float(np.linalg.norm(det_xy - ref_xy))

        # hard ok (sem tolerância) — apenas para debug
        if dist < 1e-9:
            pos_ok_hard += 1

        # soft penalty com tolerância
        if dist <= TOL_POS:
            pos_penalties.append(0.0)
        else:
            # penaliza excedente normalizado
            pos_penalties.append(min((dist - TOL_POS) / max(1e-6, (1.0 - TOL_POS)), 1.0))

    s_pos = 1.0 - (float(np.mean(pos_penalties)) if len(pos_penalties) else 0.0)
    s_pos = float(np.clip(s_pos, 0.0, 1.0))

    # ---------------------------------------------------------
    # Arestas (vizinhança) do gabarito
    # ---------------------------------------------------------
    edges, ref_dist, max_ref_edge = _build_edges_knn(centroids_ref_norm, k=3)

    # ---------------------------------------------------------
    # 2) PROXIMIDADE (igual ao seu)
    # ---------------------------------------------------------
    prox_penalties = []
    for (i, j) in edges:
        ref_d = ref_dist[(i, j)]
        pi = det_centroids_norm_por_ref[i]
        pj = det_centroids_norm_por_ref[j]
        if pi is None or pj is None:
            prox_penalties.append(1.0)
            continue
        det_d = float(np.linalg.norm(pi - pj))
        diff = abs(det_d - ref_d)
        if diff <= TOL_PROX:
            prox_penalties.append(0.0)
        else:
            prox_penalties.append(min((diff - TOL_PROX) / max_ref_edge, 1.0))

    s_prox = 1.0 - (float(np.mean(prox_penalties)) if len(prox_penalties) else 0.0)
    s_prox = float(np.clip(s_prox, 0.0, 1.0))

    # ---------------------------------------------------------
    # 3) ROTAÇÃO (melhorada): diff real em 0..180 (penaliza 180 corretamente)
    # ---------------------------------------------------------
    rot_penalties = []
    used_kp = 0

    for idx_ref in range(K):
        ang_ref = float(angles_ref_deg[idx_ref])
        ang_det = det_angles_por_ref[idx_ref]

        if ang_det is None:
            rot_penalties.append(1.0)
            continue

        if det_angsrc_por_ref[idx_ref] == "kp":
            used_kp += 1

        diff = diff_ang_real(ang_det, ang_ref)

        if diff <= TOL_ROT_DEG:
            rot_penalties.append(0.0)
        else:
            rot_penalties.append(min((diff - TOL_ROT_DEG) / (180.0 - TOL_ROT_DEG), 1.0))

    s_rot = 1.0 - (float(np.mean(rot_penalties)) if len(rot_penalties) else 0.0)
    s_rot = float(np.clip(s_rot, 0.0, 1.0))

    # ---------------------------------------------------------
    # 4) FECHAMENTO (igual ao seu)
    # ---------------------------------------------------------
    fech_penalties = []
    for (i, j) in edges:
        pi = det_centroids_norm_por_ref[i]
        pj = det_centroids_norm_por_ref[j]
        if pi is None or pj is None:
            fech_penalties.append(1.0)
            continue

        ref_vec = centroids_ref_norm[j] - centroids_ref_norm[i]
        det_vec = pj - pi
        diff_vec = float(np.linalg.norm(det_vec - ref_vec))

        if diff_vec <= TOL_FECH:
            fech_penalties.append(0.0)
        else:
            fech_penalties.append(min((diff_vec - TOL_FECH) / max_ref_edge, 1.0))

    s_fech = 1.0 - (float(np.mean(fech_penalties)) if len(fech_penalties) else 0.0)
    s_fech = float(np.clip(s_fech, 0.0, 1.0))

    # ---------------------------------------------------------
    # Nota final
    # ---------------------------------------------------------
    nota = round((W_POS * s_pos + W_PROX * s_prox + W_ROT * s_rot + W_FECH * s_fech) * 100.0, 2)

    # ---------------------------------------------------------
    # DEBUG: desenhar centróides ref/det
    # ---------------------------------------------------------
    for idx_ref in range(K):
        # ref (azul)
        cxr = centroids_ref_norm[idx_ref][0] * wg + x1g
        cyr = centroids_ref_norm[idx_ref][1] * hg + y1g
        desenhar_ponto(img_plot, cxr, cyr, (255, 0, 0), radius=6)

        # det (vermelho)
        if det_centroids_global_por_ref[idx_ref] is not None:
            cx, cy = det_centroids_global_por_ref[idx_ref]
            desenhar_ponto(img_plot, cx, cy, (0, 0, 255), radius=6)

    desenhar_texto(img_plot, f"Nota: {nota}%")

    # ---------------------------------------------------------
    # Print breakdown
    # ---------------------------------------------------------
    print(f"\n[{img_path}] ===== BREAKDOWN =====")
    print(f"  Posição:      {s_pos*100:.2f}%  (TOL_POS={TOL_POS})")
    print(f"  Proximidade:  {s_prox*100:.2f}%  (arestas {len(edges)})")
    print(f"  Rotação:      {s_rot*100:.2f}%  (keypoints usados em {used_kp}/{K})")
    print(f"  Fechamento:   {s_fech*100:.2f}%")
    print(f"  FINAL:        {nota:.2f}%")
    print(f"  Peças presentes: {presentes}/{K}\n")

    # ---------------------------------------------------------
    # Salvar imagem
    # ---------------------------------------------------------
    if salvar:
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, os.path.basename(img_path).replace(".jpg", "_avaliada.jpg"))
        cv2.imwrite(out_path, img_plot)
        print(f"[OK] Imagem avaliada salva em: {out_path}")

    return nota


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    img_correta = "correta.jpg"
    img_incorreta = "incorreta.jpg"
    img_incorreta2 = "incorreta2.jpg"

    nota_c = avaliar_montagem(img_correta, salvar=True, draw_kp=True)
    nota_i = avaliar_montagem(img_incorreta, salvar=True, draw_kp=True)
    try:
        nota_i2 = avaliar_montagem(img_incorreta2, salvar=True, draw_kp=True)
    except FileNotFoundError:
        nota_i2 = None

    print("\n===== RESULTADOS FINAIS =====")
    print(f"Correta:    {nota_c}")
    print(f"Incorreta:  {nota_i}")
    if nota_i2 is not None:
        print(f"Incorreta2: {nota_i2}")