import numpy as np
import cv2
import os
from ultralytics import YOLO

SEG_MODEL_PATH = "best-seg.pt"
GABARITO_NPZ = "gabarito_seg.npz"
OUT_DIR = "avaliacoes_seg"

# Tolerâncias
TOL_ROT_DEG = 10.0   # rotação: até 10° não penaliza
TOL_PROX = 0.08      # proximidade: tolerância em coordenadas normalizadas (0–1)
TOL_FECH = 0.06      # fechamento: tolerância em coordenadas normalizadas (0–1)

# Pesos (devem somar 1)
W_POS = 0.35
W_PROX = 0.25
W_ROT = 0.25
W_FECH = 0.15


seg_model = YOLO(SEG_MODEL_PATH)


def _assert_pesos():
    s = W_POS + W_PROX + W_ROT + W_FECH
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Pesos devem somar 1.0 (atual={s}). Ajuste W_POS/W_PROX/W_ROT/W_FECH.")


def desenhar_ponto(img, x, y, cor, radius=7):
    cv2.circle(img, (int(x), int(y)), radius, cor, -1)


def desenhar_texto(img, texto, pos=(20, 40)):
    cv2.putText(img, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    cv2.putText(img, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)


def _norm_angle_deg(angle_deg: float) -> float:
    """Normaliza ângulo para [0,360)."""
    return float((angle_deg + 360.0) % 360.0)


def calcular_angulo_mascara(mask_bin: np.ndarray) -> float:
    """Retorna ângulo DIRECIONAL em graus [0,360).

    Motivo:
      - Ângulo via momentos/PCA dá um *eixo* (0 e 180 são equivalentes).
      - Para penalizar rotação de 180°, precisamos de direção.

    Estratégia:
      - centróide da máscara
      - ponto mais distante do centróide como âncora
      - ângulo do vetor centróide -> âncora
    """
    ys, xs = np.where(mask_bin > 0)
    if len(xs) < 10:
        return 0.0

    cx = float(xs.mean())
    cy = float(ys.mean())

    dx = xs - cx
    dy = ys - cy
    idx = int(np.argmax(dx * dx + dy * dy))

    ax = float(xs[idx])
    ay = float(ys[idx])

    ang = np.degrees(np.arctan2((ay - cy), (ax - cx)))
    return _norm_angle_deg(ang)


def calcular_angulo_keypoints(kps_xy: np.ndarray) -> float:
    """Retorna ângulo DIRECIONAL em graus [0,360) a partir de keypoints.

    Observação:
      - Se você usar keypoints, a direção já existe (vetor entre 2 pontos).
      - Ainda assim, normalizamos em [0,360) para compatibilidade.
    """
    if kps_xy is None:
        return 0.0

    pts = []
    for p in kps_xy.reshape(-1, 2):
        x, y = float(p[0]), float(p[1])
        if np.isfinite(x) and np.isfinite(y) and (x != 0.0 or y != 0.0):
            pts.append((x, y))
        if len(pts) >= 2:
            break

    if len(pts) < 2:
        return 0.0

    (x1, y1), (x2, y2) = pts[0], pts[1]
    ang = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
    return _norm_angle_deg(ang)


def carregar_gabarito(npz_path: str):
    """Carrega gabarito com centroids, angles e (opcional) kps_ref."""
    data = np.load(npz_path, allow_pickle=True)

    classes_ref = data["classes"]  # (K,)
    centroids_ref_norm = data["centroids"]  # (K,2)

    kps_ref = data["kps_ref"] if "kps_ref" in data.files else None

    if "angles" in data.files:
        angles_ref = data["angles"]  # pode estar em graus ou normalizado dependendo do gerador do gabarito
        # Heurística: se estiver em [0,1], converte para graus (0–360)
        if np.nanmax(angles_ref) <= 1.0 + 1e-6:
            angles_ref_deg = angles_ref.astype(float) * 360.0
        else:
            angles_ref_deg = angles_ref.astype(float)
    else:
        angles_ref_deg = np.zeros(len(classes_ref), dtype=float)

    # Se existir kps_ref e angles não vierem (ou vierem zerados), tenta derivar dos keypoints
    if kps_ref is not None and (np.allclose(angles_ref_deg, 0.0)):
        tmp = []
        for i in range(len(classes_ref)):
            try:
                tmp.append(calcular_angulo_keypoints(np.array(kps_ref[i])))
            except Exception:
                tmp.append(0.0)
        angles_ref_deg = np.array(tmp, dtype=float)

    return classes_ref, centroids_ref_norm.astype(float), angles_ref_deg.astype(float), kps_ref


def _build_edges_knn(centroids_ref_norm: np.ndarray, k: int = 3):
    """Cria arestas (i,j) não direcionadas a partir de k vizinhos mais próximos no gabarito."""
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
    # maior distância de referência nas arestas (para normalizar penalizações)
    max_ref = 0.0
    ref_d = {}
    for (i, j) in edges:
        d = float(np.linalg.norm(centroids_ref_norm[i] - centroids_ref_norm[j]))
        ref_d[(i, j)] = d
        max_ref = max(max_ref, d)
    max_ref = max(max_ref, 1e-6)
    return edges, ref_d, max_ref


def _extrair_deteccoes(img: np.ndarray):
    """Roda segmentação e retorna:
        - img_plot (visualização do YOLO)
        - dict por classe: {class_id: {'cx_n','cy_n','cx','cy','ang_deg','conf'}}
        - bbox global (x1g,y1g,wg,hg)
        - dados brutos (results, boxes, cls, conf, masks, kps (se houver))
    """
    results = seg_model(img)[0]
    img_plot = results.plot().copy()

    if len(results.boxes) == 0 or results.masks is None:
        return img_plot, {}, None, results

    boxes = results.boxes.xyxy.cpu().numpy()
    cls = results.boxes.cls.cpu().numpy()
    conf = results.boxes.conf.cpu().numpy()
    masks = results.masks.data.cpu().numpy()  # (N,H,W)

    # keypoints (se o modelo retornar)
    kps = None
    try:
        if getattr(results, "keypoints", None) is not None and results.keypoints is not None:
            # results.keypoints.xy -> (N, n_kps, 2)
            kps = results.keypoints.xy.cpu().numpy()
    except Exception:
        kps = None

    # bbox global da maçã (normalização)
    x1_global = float(boxes[:, 0].min())
    y1_global = float(boxes[:, 1].min())
    x2_global = float(boxes[:, 2].max())
    y2_global = float(boxes[:, 3].max())

    width_global = float((x2_global - x1_global) + 1e-6)
    height_global = float((y2_global - y1_global) + 1e-6)

    det_por_classe = {}

    # Mantém 1 detecção por classe (maior conf)
    for i in range(len(cls)):
        c = int(cls[i])
        if (c not in det_por_classe) or (conf[i] > det_por_classe[c]["conf"]):
            x1, y1, x2, y2 = boxes[i]
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)

            cx_n = float((cx - x1_global) / width_global)
            cy_n = float((cy - y1_global) / height_global)

            # ângulo: preferir keypoints, senão máscara
            ang_deg = None
            if kps is not None:
                try:
                    ang_deg = calcular_angulo_keypoints(kps[i])
                except Exception:
                    ang_deg = None

            if ang_deg is None:
                mask = masks[i]
                mask_bin = (mask > 0.5).astype("uint8")
                ang_deg = calcular_angulo_mascara(mask_bin)

            det_por_classe[c] = {
                "cx": cx,
                "cy": cy,
                "cx_n": cx_n,
                "cy_n": cy_n,
                "ang_deg": float(ang_deg),
                "conf": float(conf[i]),
            }

    bbox = (x1_global, y1_global, width_global, height_global)
    return img_plot, det_por_classe, bbox, results


def _diff_ang_deg(a: float, b: float) -> float:
    """Diferença angular DIRECIONAL em graus (circular) em [0,180].

    - Usa 360° como período (agora 0 e 180 NÃO são equivalentes).
    - Retorna o menor arco entre os dois ângulos.
    """
    d = abs(float(a) - float(b)) % 360.0
    d = min(d, 360.0 - d)
    return float(d)


def avaliar_montagem_seg(img_path: str, salvar: bool = True) -> float:
    """Avalia montagem com 4 scores: posição, proximidade, rotação, fechamento."""
    _assert_pesos()

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Não consegui carregar a imagem: {img_path}")

    classes_ref, centroids_ref_norm, angles_ref_deg, kps_ref = carregar_gabarito(GABARITO_NPZ)
    K = len(classes_ref)

    img_plot, det_por_classe, bbox, results = _extrair_deteccoes(img)

    if bbox is None:
        print(f"[{img_path}] Nenhuma peça segmentada (ou sem máscaras).")
        nota = 0.0
        if salvar:
            os.makedirs(OUT_DIR, exist_ok=True)
            out_img = os.path.join(OUT_DIR, os.path.basename(img_path))
            cv2.imwrite(out_img, img)
        return nota

    x1g, y1g, wg, hg = bbox

    # index do gabarito por classe -> idx ref
    idx_ref_by_class = {int(classes_ref[i]): i for i in range(K)}

    # centroids previstos alinhados com ref (K,2), e flags presentes
    centroids_pred_norm = np.zeros((K, 2), dtype=float)
    angles_pred_deg = np.zeros((K,), dtype=float)
    present = np.zeros((K,), dtype=bool)

    for c, d in det_por_classe.items():
        if c in idx_ref_by_class:
            i = idx_ref_by_class[c]
            centroids_pred_norm[i, 0] = d["cx_n"]
            centroids_pred_norm[i, 1] = d["cy_n"]
            angles_pred_deg[i] = d["ang_deg"]
            present[i] = True

    # --------------------
    # Score 1) Posição
    # --------------------
    pos_scores = []
    for i in range(K):
        if not present[i]:
            pos_scores.append(0.0)
            continue
        dist = float(np.linalg.norm(centroids_pred_norm[i] - centroids_ref_norm[i]))
        s = max(0.0, 1.0 - (dist / max(TOL_PROX, 1e-6)))  # reaproveita escala de tolerância semelhante
        pos_scores.append(s)

    score_pos = float(np.mean(pos_scores)) if len(pos_scores) else 0.0

    # --------------------
    # Score 2) Proximidade (arestas kNN)
    # --------------------
    edges, ref_d, max_ref = _build_edges_knn(centroids_ref_norm, k=3)

    prox_scores = []
    for (i, j) in edges:
        if not (present[i] and present[j]):
            prox_scores.append(0.0)
            continue

        d_pred = float(np.linalg.norm(centroids_pred_norm[i] - centroids_pred_norm[j]))
        d_ref = float(ref_d[(i, j)])

        # erro relativo em relação ao maior ref (normalização)
        err = abs(d_pred - d_ref)

        # penalização com tolerância
        s = max(0.0, 1.0 - (err / max(TOL_PROX, 1e-6)))
        prox_scores.append(s)

    score_prox = float(np.mean(prox_scores)) if len(prox_scores) else 0.0

    # --------------------
    # Score 3) Rotação
    # --------------------
    rot_scores = []
    for i in range(K):
        if not present[i]:
            rot_scores.append(0.0)
            continue

        d = _diff_ang_deg(angles_pred_deg[i], angles_ref_deg[i])  # ✅ agora 360-circular
        s = max(0.0, 1.0 - (d / max(TOL_ROT_DEG, 1e-6)))
        rot_scores.append(s)

    score_rot = float(np.mean(rot_scores)) if len(rot_scores) else 0.0

    # --------------------
    # Score 4) Fechamento
    # --------------------
    # fechamento: compara "perímetro" (ciclo) aproximando por ordenar por ângulo polar ao redor do centro
    # (mantém seu espírito: consistência global)
    fech_scores = []
    present_idxs = [i for i in range(K) if present[i]]
    if len(present_idxs) >= 3:
        # centro médio
        c0 = centroids_pred_norm[present_idxs].mean(axis=0)

        # ângulo polar para ordenar
        angs = []
        for i in present_idxs:
            v = centroids_pred_norm[i] - c0
            ang = np.degrees(np.arctan2(v[1], v[0]))
            angs.append((ang, i))
        angs.sort(key=lambda x: x[0])
        order = [i for _, i in angs]

        # soma de distâncias em ciclo (pred) e ref
        def cycle_len(points):
            L = 0.0
            for a in range(len(points)):
                i = points[a]
                j = points[(a + 1) % len(points)]
                L += float(np.linalg.norm(points[i] - points[j]))
            return L

        # cria arrays indexáveis por idx (K,2)
        pred_pts = centroids_pred_norm
        ref_pts = centroids_ref_norm

        # comprimento apenas nos mesmos índices presentes (ordem do pred, aplicada no ref também)
        L_pred = 0.0
        L_ref = 0.0
        for a in range(len(order)):
            i = order[a]
            j = order[(a + 1) % len(order)]
            L_pred += float(np.linalg.norm(pred_pts[i] - pred_pts[j]))
            L_ref += float(np.linalg.norm(ref_pts[i] - ref_pts[j]))

        err = abs(L_pred - L_ref)
        score_fech = max(0.0, 1.0 - (err / max(TOL_FECH, 1e-6)))
    else:
        score_fech = 0.0

    # --------------------
    # Final
    # --------------------
    score_final = (W_POS * score_pos) + (W_PROX * score_prox) + (W_ROT * score_rot) + (W_FECH * score_fech)

    # --------------------
    # Relatório/Debug
    # --------------------
    print(f"\n[{os.path.basename(img_path)}] ===== BREAKDOWN =====")
    print(f"  Posição:      {score_pos*100:.2f}%")
    print(f"  Proximidade:  {score_prox*100:.2f}%  (arestas {len(edges)})")
    print(f"  Rotação:      {score_rot*100:.2f}%   (tol {TOL_ROT_DEG:.1f}°)")
    print(f"  Fechamento:   {score_fech*100:.2f}%")
    print(f"  FINAL:        {score_final*100:.2f}%")
    print(f"  Peças presentes: {int(present.sum())}/{K}")

    # --------------------
    # Salvar artefatos
    # --------------------
    if salvar:
        os.makedirs(OUT_DIR, exist_ok=True)

        # desenha centróides previstos e texto de score
        out_vis = img_plot.copy()

        for i in range(K):
            if not present[i]:
                continue
            cx = x1g + centroids_pred_norm[i, 0] * wg
            cy = y1g + centroids_pred_norm[i, 1] * hg
            desenhar_ponto(out_vis, cx, cy, (0, 255, 0), radius=7)

        desenhar_texto(out_vis, f"FINAL: {score_final*100:.1f}%", pos=(20, 40))

        out_img = os.path.join(OUT_DIR, os.path.basename(img_path))
        cv2.imwrite(out_img, out_vis)

    return float(score_final)


if __name__ == "__main__":
    # Exemplos
    avaliar_montagem_seg("correta.jpg", salvar=True)
    avaliar_montagem_seg("incorreta.jpg", salvar=True)
    avaliar_montagem_seg("incorreta2.jpg", salvar=True)
