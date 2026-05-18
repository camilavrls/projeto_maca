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

    resultado = seg_model(img_rgb, conf=0.10)[0]

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

def calcular_centroide_normalizado(masks, boxes, imagem_shape) -> tuple:
    x_min = np.min(boxes[:, 0])
    y_min = np.min(boxes[:, 1])
    x_max = np.max(boxes[:, 2])
    y_max = np.max(boxes[:, 3])
    
    largura_total = x_max - x_min
    altura_total = y_max - y_min
    altura_img, largura_img = imagem_shape[:2]
    altura_mask, largura_mask = masks.shape[1:3]
    escala_x = largura_img / largura_mask
    escala_y = altura_img / altura_mask
    
    centroides_abs = []
    centroides_normalizados = []
    
    for i, mascara in enumerate(masks):
        y_pixels, x_pixels = np.where(mascara == 1)

        x_pixels = x_pixels * escala_x
        y_pixels = y_pixels * escala_y
        
        cx_abs = np.mean(x_pixels)
        cy_abs = np.mean(y_pixels)

        centroides_abs.append([cx_abs, cy_abs])
        
        cx_norm = (cx_abs - x_min) / largura_total
        cy_norm = (cy_abs - y_min) / altura_total
        
        centroides_normalizados.append([cx_norm, cy_norm])

    return np.array(centroides_abs), np.array(centroides_normalizados)

def calcular_angulo_orientacao(masks, centroides_abs, imagem_shape) -> np.ndarray:
    angulos = []
    altura_img, largura_img = imagem_shape[:2]
    altura_mask, largura_mask = masks.shape[1:3]
    escala_x = largura_img / largura_mask
    escala_y = altura_img / altura_mask
    
    for i, mascara in enumerate(masks):
        
        y_pixels, x_pixels = np.where(mascara == 1)

        x_pixels = x_pixels * escala_x
        y_pixels = y_pixels * escala_y
        
        pontos = np.column_stack([x_pixels, y_pixels])
        pontos_centralizados = pontos - pontos.mean(axis=0)
        matriz_cov = np.cov(pontos_centralizados.T)
        valores, vetores = np.linalg.eigh(matriz_cov)
        vetor_principal = vetores[:, np.argmax(valores)]

        angulo_deg = np.degrees(np.arctan2(vetor_principal[1], vetor_principal[0]))
        if angulo_deg < 0:
            angulo_deg += 180
        
        angulos.append(angulo_deg)
    
    return np.array(angulos)

def calcular_relacoes_pares(centroides_norm, classes) -> dict:

    indice_classes = {classe: i for i, classe in enumerate(classes)}

    idx_esquerda = indice_classes.get("Esquerda")
    idx_direita_cima = indice_classes.get("Direita_cima")
    idx_direita_baixo = indice_classes.get("Direita_baixo")
    
    if None in [idx_esquerda, idx_direita_cima, idx_direita_baixo]:
        raise ValueError("Nem todas as peças foram detectadas!")
    
    relacoes = {}
    
    # PAR 1: Esquerda ↔ Direita_cima
    cx_esq = centroides_norm[idx_esquerda, 0]
    cy_esq = centroides_norm[idx_esquerda, 1]
    cx_cima = centroides_norm[idx_direita_cima, 0]
    cy_cima = centroides_norm[idx_direita_cima, 1]
    
    dx_esq_cima = cx_cima - cx_esq
    dy_esq_cima = cy_cima - cy_esq
    
    dist_esq_cima = np.sqrt(dx_esq_cima**2 + dy_esq_cima**2)
    
    ang_rad_esq_cima = np.arctan2(dy_esq_cima, dx_esq_cima)
    ang_deg_esq_cima = np.degrees(ang_rad_esq_cima)
    if ang_deg_esq_cima < 0:
        ang_deg_esq_cima += 360
    
    relacoes["esquerda_cima"] = {
        "distancia": float(dist_esq_cima),
        "angulo": float(ang_deg_esq_cima)
    }
    
    # PAR 2: Esquerda ↔ Direita_baixo 
    cx_baixo = centroides_norm[idx_direita_baixo, 0]
    cy_baixo = centroides_norm[idx_direita_baixo, 1]
    
    dx_esq_baixo = cx_baixo - cx_esq
    dy_esq_baixo = cy_baixo - cy_esq
    
    dist_esq_baixo = np.sqrt(dx_esq_baixo**2 + dy_esq_baixo**2)
    
    ang_rad_esq_baixo = np.arctan2(dy_esq_baixo, dx_esq_baixo)
    ang_deg_esq_baixo = np.degrees(ang_rad_esq_baixo)
    if ang_deg_esq_baixo < 0:
        ang_deg_esq_baixo += 360
    
    relacoes["esquerda_baixo"] = {
        "distancia": float(dist_esq_baixo),
        "angulo": float(ang_deg_esq_baixo)
    }
    
    # PAR 3: Direita_cima ↔ Direita_baixo 
    dx_cima_baixo = cx_baixo - cx_cima
    dy_cima_baixo = cy_baixo - cy_cima
    
    dist_cima_baixo = np.sqrt(dx_cima_baixo**2 + dy_cima_baixo**2)
    
    ang_rad_cima_baixo = np.arctan2(dy_cima_baixo, dx_cima_baixo)
    ang_deg_cima_baixo = np.degrees(ang_rad_cima_baixo)
    if ang_deg_cima_baixo < 0:
        ang_deg_cima_baixo += 360
    
    relacoes["cima_baixo"] = {
        "distancia": float(dist_cima_baixo),
        "angulo": float(ang_deg_cima_baixo)
    }
    
    return relacoes

def salvar_gabarito(centroides_norm, angulos_abs, relacoes_pares, classes, npz_path: str) -> None:

    pares_esq_cima_dist = relacoes_pares["esquerda_cima"]["distancia"]
    pares_esq_cima_ang = relacoes_pares["esquerda_cima"]["angulo"]
    
    pares_esq_baixo_dist = relacoes_pares["esquerda_baixo"]["distancia"]
    pares_esq_baixo_ang = relacoes_pares["esquerda_baixo"]["angulo"]
    
    pares_cima_baixo_dist = relacoes_pares["cima_baixo"]["distancia"]
    pares_cima_baixo_ang = relacoes_pares["cima_baixo"]["angulo"]
    
    np.savez_compressed(
        npz_path,
        classes=classes,
        centroides_norm=centroides_norm,
        angulos_abs=angulos_abs,
        pares_esq_cima_dist=pares_esq_cima_dist,
        pares_esq_cima_ang=pares_esq_cima_ang,
        pares_esq_baixo_dist=pares_esq_baixo_dist,
        pares_esq_baixo_ang=pares_esq_baixo_ang,
        pares_cima_baixo_dist=pares_cima_baixo_dist,
        pares_cima_baixo_ang=pares_cima_baixo_ang
    )
    
    print(f"Gabarito salvo em: {npz_path}")

def carregar_gabarito(npz_path: str) -> dict:
    dados = np.load(npz_path, allow_pickle=True)
    
    relacoes_pares = {
        "esquerda_cima": {
            "distancia": float(dados["pares_esq_cima_dist"]),
            "angulo": float(dados["pares_esq_cima_ang"])
        },
        "esquerda_baixo": {
            "distancia": float(dados["pares_esq_baixo_dist"]),
            "angulo": float(dados["pares_esq_baixo_ang"])
        },
        "cima_baixo": {
            "distancia": float(dados["pares_cima_baixo_dist"]),
            "angulo": float(dados["pares_cima_baixo_ang"])
        }
    }

    return {
        "classes": list(dados["classes"]),
        "centroides_norm": dados["centroides_norm"],
        "angulos_abs": dados["angulos_abs"],
        "relacoes_pares": relacoes_pares
    }

if __name__ == "__main__":
    print("=" * 60)
    print("GERANDO GABARITO DE SEGMENTAÇÃO")
    print("=" * 60)
    
    print("\n=== ETAPA 1: SEGMENTAÇÃO 🎯 ===")
    resultado = processar_imagem("correta.jpg")
    print(f"Classes detectadas: {resultado['classes']}")
    print(f"Confiança: {resultado['confiancas']}")
    
    print("\n=== ETAPA 2A: CENTRÓIDES NORMALIZADOS ===")
    centroides_abs, centroides_norm = calcular_centroide_normalizado(
        resultado["masks"],
        resultado["boxes"],
        resultado["imagem"].shape
    )
    print("Centróides Normalizados:")
    for i, classe in enumerate(resultado["classes"]):
        print(f"  {classe}: ({centroides_norm[i, 0]:.4f}, {centroides_norm[i, 1]:.4f})")
    
    print("\n=== ETAPA 2B: ÂNGULOS DE ORIENTAÇÃO 📐 ===")
    angulos = calcular_angulo_orientacao(
        resultado["masks"],
        centroides_abs,
        resultado["imagem"].shape
    )
    print("Ângulos de Orientação (0-360°):")
    for i, classe in enumerate(resultado["classes"]):
        print(f"  {classe}: {angulos[i]:.2f}°")
    
    print("\n=== ETAPA 3: RELAÇÕES ENTRE PARES 🔗 ===")
    relacoes = calcular_relacoes_pares(centroides_norm, resultado["classes"])
    print("Relações entre Pares:")
    for par, dados in relacoes.items():
        print(f"  {par}:")
        print(f"    - Distância: {dados['distancia']:.4f}")
        print(f"    - Ângulo Relativo: {dados['angulo']:.2f}°")
    
    print("\n=== ETAPA 4: PERSISTÊNCIA 💾 ===")
    salvar_gabarito(
        centroides_norm,
        angulos,
        relacoes,
        resultado["classes"],
        GABARITO_NPZ
    )
    
    print("\n=== VERIFICAÇÃO: Carregando Gabarito ===")
    gabarito = carregar_gabarito(GABARITO_NPZ)
    print("✅ Gabarito carregado com sucesso!")
    print(f"Classes: {gabarito['classes']}")
    print(f"Centróides normalizados:\n{gabarito['centroides_norm']}")
    print(f"Ângulos absolutos:\n{gabarito['angulos_abs']}")
    print(f"Relações entre pares:")
    for par, dados in gabarito['relacoes_pares'].items():
        print(f"  {par}: dist={dados['distancia']:.4f}, ang={dados['angulo']:.2f}°")
    
    print("\n" + "=" * 60)
    print("✅ GABARITO GERADO COM SUCESSO!")
    print("=" * 60)
