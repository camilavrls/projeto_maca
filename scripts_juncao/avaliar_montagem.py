import numpy as np
import cv2
from ultralytics import YOLO

GABARITO_PATH = "gabarito_seg.npz"
SEG_MODEL_PATH = "../best-seg.pt"
TOL_DIST_FRAC = 0.20      # 20% de tolerância na distância
TOL_ANG_DEG = 50.0        # 25° de tolerância no ângulo relativo
TOL_ROT_DEG = 30.0        # 30° de tolerância na rotação

seg_model = YOLO(SEG_MODEL_PATH)


def carregar_gabarito(npz_path):
    dados = np.load(npz_path, allow_pickle=True)
    return {
        "classes": list(dados["classes"]),
        "centroides_norm": dados["centroides_norm"],
        "angulos_abs": dados["angulos_abs"],
        "relacoes_pares": {
            "esquerda_cima": {"dist": float(dados["pares_esq_cima_dist"]), "ang": float(dados["pares_esq_cima_ang"])},
            "esquerda_baixo": {"dist": float(dados["pares_esq_baixo_dist"]), "ang": float(dados["pares_esq_baixo_ang"])},
            "cima_baixo": {"dist": float(dados["pares_cima_baixo_dist"]), "ang": float(dados["pares_cima_baixo_ang"])}
        }
    }


def segmentar_imagem(img_path, gabarito):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultado = seg_model(img_rgb)[0]
    
    boxes = resultado.boxes.xyxy.cpu().numpy()
    masks = resultado.masks.data.cpu().numpy()
    classes = resultado.boxes.cls.cpu().numpy()

    nome_classes = [seg_model.names[int(c)] for c in classes]

    x_min, y_min = np.min(boxes[:, 0]), np.min(boxes[:, 1])
    x_max, y_max = np.max(boxes[:, 2]), np.max(boxes[:, 3])
    w_total, h_total = x_max - x_min, y_max - y_min
    
    centroides_norm = []
    centroides_abs = []
    for mascara in masks:
        y_pix, x_pix = np.where(mascara == 1)
        cx_abs, cy_abs = np.mean(x_pix), np.mean(y_pix)
        centroides_abs.append([cx_abs, cy_abs])
        cx_norm = (cx_abs - x_min) / w_total
        cy_norm = (cy_abs - y_min) / h_total
        centroides_norm.append([cx_norm, cy_norm])
    
    angulos = []
    for i, mascara in enumerate(masks):
        y_pix, x_pix = np.where(mascara == 1)
        cx, cy = centroides_abs[i]
        dist = np.sqrt((x_pix - cx)**2 + (y_pix - cy)**2)
        x_d, y_d = x_pix[np.argmax(dist)], y_pix[np.argmax(dist)]
        ang = np.degrees(np.arctan2(y_d - cy, x_d - cx))
        angulos.append(ang + 360 if ang < 0 else ang)
    
    return {
        "classes": nome_classes,
        "centroides_norm": np.array(centroides_norm),
        "angulos_abs": np.array(angulos)
    }


def calcular_distancia(c1, c2):
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def calcular_angulo(c1, c2):
    dx, dy = c2[0] - c1[0], c2[1] - c1[1]
    ang = np.degrees(np.arctan2(dy, dx))
    return ang + 360 if ang < 0 else ang


def diferenca_angular(a1, a2):
    diff = abs(a1 - a2)
    return min(diff, 360 - diff)


def avaliar_par(nome_par, gabarito_atual, montagem_atual):
    
    idx_map = {"Esquerda": 0, "Direita_cima": 1, "Direita_baixo": 2}
    
    if nome_par == "esquerda_cima":
        idx_a, idx_b = idx_map["Esquerda"], idx_map["Direita_cima"]
    elif nome_par == "esquerda_baixo":
        idx_a, idx_b = idx_map["Esquerda"], idx_map["Direita_baixo"]
    else:  # cima_baixo
        idx_a, idx_b = idx_map["Direita_cima"], idx_map["Direita_baixo"]
    
    c_a_ref = gabarito_atual["centroides_norm"][idx_a]
    c_b_ref = gabarito_atual["centroides_norm"][idx_b]
    c_a_novo = montagem_atual["centroides_norm"][idx_a]
    c_b_novo = montagem_atual["centroides_norm"][idx_b]
    
    ang_a_ref = gabarito_atual["angulos_abs"][idx_a]
    ang_b_ref = gabarito_atual["angulos_abs"][idx_b]
    ang_a_novo = montagem_atual["angulos_abs"][idx_a]
    ang_b_novo = montagem_atual["angulos_abs"][idx_b]
    
    # Critério 1: Distância
    dist_ref = calcular_distancia(c_a_ref, c_b_ref)
    dist_novo = calcular_distancia(c_a_novo, c_b_novo)
    erro_dist = abs(dist_novo - dist_ref) / dist_ref
    crit1_ok = erro_dist <= TOL_DIST_FRAC
    
    # Critério 2: Ângulo relativo
    ang_rel_ref = calcular_angulo(c_a_ref, c_b_ref)
    ang_rel_novo = calcular_angulo(c_a_novo, c_b_novo)
    erro_ang = diferenca_angular(ang_rel_novo, ang_rel_ref)
    crit2_ok = erro_ang <= TOL_ANG_DEG
    
    # Critério 3: Rotação absoluta
    erro_rot_a = diferenca_angular(ang_a_novo, ang_a_ref)
    erro_rot_b = diferenca_angular(ang_b_novo, ang_b_ref)
    crit3_ok = (erro_rot_a <= TOL_ROT_DEG) and (erro_rot_b <= TOL_ROT_DEG)
    
    return {
        "par": nome_par,
        "crit1": crit1_ok,
        "crit1_valor": erro_dist,
        "crit2": crit2_ok,
        "crit2_valor": erro_ang,
        "crit3": crit3_ok,
        "crit3_valor_a": erro_rot_a,
        "crit3_valor_b": erro_rot_b,
        "passou": crit1_ok and crit2_ok and crit3_ok
    }


def avaliar_montagem(img_path, gabarito_path=GABARITO_PATH):
    """Avalia montagem completa."""
    
    print("\n" + "="*60)
    print("AVALIANDO MONTAGEM")
    print("="*60)
    
    print("\n=== ETAPA 1: CARREGANDO GABARITO ===")
    gabarito = carregar_gabarito(gabarito_path)
    print(f"✅ Gabarito carregado: {gabarito['classes']}")
    
    print("\n=== ETAPA 2: SEGMENTANDO IMAGEM ===")
    montagem = segmentar_imagem(img_path, gabarito)
    print(f"✅ Imagem segmentada: {montagem['classes']}")
    
    print("\n=== ETAPA 3: AVALIANDO PARES ===")
    pares = ["esquerda_cima", "esquerda_baixo", "cima_baixo"]
    resultados = []
    nota = 0
    
    for par in pares:
        resultado = avaliar_par(par, gabarito, montagem)
        resultados.append(resultado)
        
        status = "✅" if resultado["passou"] else "❌"
        print(f"\n{status} {par.upper()}")
        print(f"   Crit1 (Dist): {resultado['crit1']} (erro={resultado['crit1_valor']:.2%})")
        print(f"   Crit2 (Ang):  {resultado['crit2']} (erro={resultado['crit2_valor']:.1f}°)")
        print(f"   Crit3 (Rot):  {resultado['crit3']} (erro_a={resultado['crit3_valor_a']:.1f}°, erro_b={resultado['crit3_valor_b']:.1f}°)")
        
        if resultado["passou"]:
            nota += 1
    
    print("\n" + "="*60)
    print(f"NOTA FINAL: {nota}/3")
    print("="*60 + "\n")
    
    return {"nota": nota, "detalhes": resultados}


if __name__ == "__main__":
    resultado = avaliar_montagem("incorreta2.jpg")
