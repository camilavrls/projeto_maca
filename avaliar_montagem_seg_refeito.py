import numpy as np
import cv2
import os
from ultralytics import YOLO

# TODO: o algoritmo está considerando 180 como 0 e não penalizando a criança no momento em que rotaciona por completo, 
# precisa ser corrigido

# =========================
# Parâmetros (ajuste aqui)
# =========================
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

# =========================

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
    # normaliza para [0,180)
    return float((angle_deg + 180.0) % 180.0)


def calcular_angulo_mascara(mask_bin: np.ndarray) -> float:
    """Retorna ângulo em graus [0, 180)."""
    M = cv2.moments(mask_bin)
    mu11 = M.get("mu11", 0.0)
    mu20 = M.get("mu20", 0.0)
    mu02 = M.get("mu02", 0.0)

    if (mu20 - mu02) == 0 and mu11 == 0:
        angle_deg = 0.0
    else:
        angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
        angle_deg = np.degrees(angle)

    return _norm_angle_deg(angle_deg)


def calcular_angulo_keypoints(kps_xy: np.ndarray) -> float:
    """Retorna ângulo em graus [0,180) a partir de keypoints (preferência).

    Estratégia simples:
      - pega os 2 primeiros keypoints válidos (x,y) e calcula o vetor entre eles.
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
        # Heurística: se estiver em [0,1], converte para graus (0–180)
        if np.nanmax(angles_ref) <= 1.0 + 1e-6:
            angles_ref_deg = angles_ref.astype(float) * 180.0
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
            ang_deg = 0.0
            if kps is not None:
                try:
                    ang_deg = calcular_angulo_keypoints(kps[i])
                except Exception:
                    ang_deg = 0.0

            if ang_deg == 0.0:
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
    """Diferença angular em graus em [0, 90] por simetria em 180."""
    d = abs(float(a) - float(b)) % 180.0
    d = min(d, 180.0 - d)
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
            out_path = os.path.join(OUT_DIR, os.path.basename(img_path).replace(".jpg", "_seg_avaliada.jpg"))
            desenhar_texto(img_plot, f"Nota: {nota}%")
            cv2.imwrite(out_path, img_plot)
        return nota

    x1_global, y1_global, width_global, height_global = bbox

    # -------------------------
    # 1) POSIÇÃO (0/1, sem tolerância)
    # -------------------------
    # Para cada peça detectada, achamos "slot" pela referência mais próxima.
    # Acertou se o slot for o slot correto (idx_ref).
    pos_ok = 0
    presentes = 0

    det_centroids_norm_por_ref = [None] * K
    det_centroids_global_por_ref = [None] * K
    det_angles_por_ref = [None] * K

    for idx_ref, c_ref in enumerate(classes_ref):
        c_ref_int = int(c_ref)
        if c_ref_int not in det_por_classe:
            continue
        presentes += 1
        d = det_por_classe[c_ref_int]
        det_centroids_norm_por_ref[idx_ref] = np.array([d["cx_n"], d["cy_n"]], dtype=float)
        det_centroids_global_por_ref[idx_ref] = (d["cx"], d["cy"])
        det_angles_por_ref[idx_ref] = float(d["ang_deg"])

        # slot pelo centróide mais próximo no gabarito
        dist_all = np.linalg.norm(centroids_ref_norm - det_centroids_norm_por_ref[idx_ref], axis=1)
        slot_pred = int(np.argmin(dist_all))
        if slot_pred == idx_ref:
            pos_ok += 1

    s_pos = (pos_ok / K) if K > 0 else 0.0

    # -------------------------
    # Arestas (vizinhança) a partir do gabarito
    # -------------------------
    edges, ref_dist, max_ref_edge = _build_edges_knn(centroids_ref_norm, k=3)

    # -------------------------
    # 2) PROXIMIDADE (com tolerância)
    # -------------------------
    # Compara distâncias entre peças vizinhas (edges) com o gabarito.
    prox_penalties = []
    for (i, j) in edges:
        ref_d = ref_dist[(i, j)]
        pi = det_centroids_norm_por_ref[i]
        pj = det_centroids_norm_por_ref[j]
        if pi is None or pj is None:
            # ausência de peça -> penaliza
            prox_penalties.append(1.0)
            continue
        det_d = float(np.linalg.norm(pi - pj))
        diff = abs(det_d - ref_d)
        if diff <= TOL_PROX:
            prox_penalties.append(0.0)
        else:
            # penaliza o excedente, normalizado
            prox_penalties.append(min((diff - TOL_PROX) / max_ref_edge, 1.0))

    s_prox = 1.0 - (float(np.mean(prox_penalties)) if len(prox_penalties) else 0.0)
    s_prox = float(np.clip(s_prox, 0.0, 1.0))

    # -------------------------
    # 3) ROTAÇÃO (com tolerância)
    # -------------------------
    rot_penalties = []
    for idx_ref in range(K):
        ang_ref = float(angles_ref_deg[idx_ref])
        ang_det = det_angles_por_ref[idx_ref]
        if ang_det is None:
            rot_penalties.append(1.0)
            continue

        diff = _diff_ang_deg(ang_det, ang_ref)
        if diff <= TOL_ROT_DEG:
            rot_penalties.append(0.0)
        else:
            rot_penalties.append(min((diff - TOL_ROT_DEG) / (180.0 - TOL_ROT_DEG), 1.0))

    s_rot = 1.0 - (float(np.mean(rot_penalties)) if len(rot_penalties) else 0.0)
    s_rot = float(np.clip(s_rot, 0.0, 1.0))

    # -------------------------
    # 4) FECHAMENTO (com tolerância)
    # -------------------------
    # Mede "desalinhamento/gap" como variação do vetor relativo entre vizinhos.
    # (dx,dy) detectado vs (dx,dy) referência.
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

    # -------------------------
    # Combinar e imprimir breakdown
    # -------------------------
    nota = round((W_POS * s_pos + W_PROX * s_prox + W_ROT * s_rot + W_FECH * s_fech) * 100.0, 2)

    print(f"\n[{img_path}] ===== BREAKDOWN =====")
    print(f"  Posição:      {s_pos*100:.2f}%  (acertos {pos_ok}/{K})")
    print(f"  Proximidade:  {s_prox*100:.2f}%  (arestas {len(edges)})")
    print(f"  Rotação:      {s_rot*100:.2f}%")
    print(f"  Fechamento:   {s_fech*100:.2f}%")
    print(f"  FINAL:        {nota:.2f}%")
    print(f"  Peças presentes: {presentes}/{K}\n")

    # Visualização: centróides detectados (vermelho) e slots (azul)
    for idx_ref in range(K):
        # referência (azul)
        cxr = centroids_ref_norm[idx_ref][0] * width_global + x1_global
        cyr = centroids_ref_norm[idx_ref][1] * height_global + y1_global
        desenhar_ponto(img_plot, cxr, cyr, (255, 0, 0))

        # detectado (vermelho)
        if det_centroids_global_por_ref[idx_ref] is not None:
            cx, cy = det_centroids_global_por_ref[idx_ref]
            desenhar_ponto(img_plot, cx, cy, (0, 0, 255))

    desenhar_texto(img_plot, f"Nota: {nota}%")

    if salvar:
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, os.path.basename(img_path).replace(".jpg", "_seg_avaliada.jpg"))
        cv2.imwrite(out_path, img_plot)
        print(f"[OK] Imagem avaliada salva em: {out_path}")

    return nota


if __name__ == "__main__":
    # Exemplos (ajuste nomes/paths conforme seu dataset)
    img_correta = "correta.jpg"
    img_incorreta = "incorreta.jpg"
    img_incorreta2 = "incorreta2.jpg"  # opcional

    nota_c = avaliar_montagem_seg(img_correta)
    nota_i = avaliar_montagem_seg(img_incorreta)
    try:
        nota_i2 = avaliar_montagem_seg(img_incorreta2)
    except FileNotFoundError:
        nota_i2 = None

    print("\n===== RESULTADOS FINAIS =====")
    print(f"Correta:   {nota_c}")
    print(f"Incorreta: {nota_i}")
    if nota_i2 is not None:
        print(f"Incorreta2: {nota_i2}")
