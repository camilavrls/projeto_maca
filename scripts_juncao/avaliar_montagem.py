import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path
from openpyxl import Workbook
import re

GABARITO_PATH = "gabarito_seg.npz"
SEG_MODEL_PATH = "../best-seg.pt"
IMAGENS_AVALIACAO_DIR = "imagens_avaliacao"
EXCEL_OUTPUT = "notas_montagens.xlsx"
CONF_MIN = 0.10
TOL_DIST_FRAC = 0.20      # 20% de tolerância na distância
TOL_ANG_DEG = 50.0        # 25° de tolerância no ângulo relativo
TOL_ROT_DEG = 30.0        # 30° de tolerância na rotação
TOL_CONTATO_MIN = 0.25    # contato mínimo entre as máscaras das peças
CLASSES_ESPERADAS = {"Esquerda", "Direita_cima", "Direita_baixo"}
EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

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

    if img is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {img_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultado = seg_model(img_rgb, conf=CONF_MIN)[0]

    if resultado.masks is None or len(resultado.boxes) == 0:
        return {
            "classes": [],
            "centroides_norm": np.array([]),
            "angulos_abs": np.array([]),
            "confiancas": np.array([]),
            "masks_img": np.array([]),
            "duplicadas": []
        }
    
    boxes = resultado.boxes.xyxy.cpu().numpy()
    masks = resultado.masks.data.cpu().numpy()
    classes = resultado.boxes.cls.cpu().numpy()
    confiancas = resultado.boxes.conf.cpu().numpy()

    nome_classes = [seg_model.names[int(c)] for c in classes]
    boxes, masks, nome_classes, confiancas, duplicadas = deduplicar_por_confianca(
        boxes,
        masks,
        nome_classes,
        confiancas
    )

    if duplicadas:
        print("⚠️  Detecções duplicadas removidas:")
        for duplicada in duplicadas:
            print(f"   - {duplicada}")

    x_min, y_min = np.min(boxes[:, 0]), np.min(boxes[:, 1])
    x_max, y_max = np.max(boxes[:, 2]), np.max(boxes[:, 3])
    w_total, h_total = x_max - x_min, y_max - y_min
    h_img, w_img = img.shape[:2]
    h_mask, w_mask = masks.shape[1:3]
    escala_x = w_img / w_mask
    escala_y = h_img / h_mask
    
    centroides_norm = []
    centroides_abs = []
    masks_img = []
    for mascara in masks:
        mascara_img = cv2.resize(
            mascara.astype(np.uint8),
            (w_img, h_img),
            interpolation=cv2.INTER_NEAREST
        )
        masks_img.append(mascara_img)

        y_pix, x_pix = np.where(mascara == 1)
        x_pix = x_pix * escala_x
        y_pix = y_pix * escala_y
        cx_abs, cy_abs = np.mean(x_pix), np.mean(y_pix)
        centroides_abs.append([cx_abs, cy_abs])
        cx_norm = (cx_abs - x_min) / w_total
        cy_norm = (cy_abs - y_min) / h_total
        centroides_norm.append([cx_norm, cy_norm])
    
    angulos = []
    for i, mascara in enumerate(masks):
        y_pix, x_pix = np.where(mascara == 1)
        x_pix = x_pix * escala_x
        y_pix = y_pix * escala_y
        pontos = np.column_stack([x_pix, y_pix])
        pontos_centralizados = pontos - pontos.mean(axis=0)
        matriz_cov = np.cov(pontos_centralizados.T)
        valores, vetores = np.linalg.eigh(matriz_cov)
        vetor_principal = vetores[:, np.argmax(valores)]

        ang = np.degrees(np.arctan2(vetor_principal[1], vetor_principal[0]))
        angulos.append(ang + 180 if ang < 0 else ang)
    
    return {
        "classes": nome_classes,
        "centroides_norm": np.array(centroides_norm),
        "angulos_abs": np.array(angulos),
        "confiancas": np.array(confiancas),
        "masks_img": np.array(masks_img),
        "duplicadas": duplicadas
    }


def deduplicar_por_confianca(boxes, masks, nome_classes, confiancas):
    melhor_por_classe = {}
    duplicadas = []

    for i, (classe, confianca) in enumerate(zip(nome_classes, confiancas)):
        if classe not in melhor_por_classe:
            melhor_por_classe[classe] = i
            continue

        idx_atual = melhor_por_classe[classe]
        if confianca > confiancas[idx_atual]:
            duplicadas.append(
                f"{classe}: descartada conf={confiancas[idx_atual]:.4f}, mantida conf={confianca:.4f}"
            )
            melhor_por_classe[classe] = i
        else:
            duplicadas.append(
                f"{classe}: descartada conf={confianca:.4f}, mantida conf={confiancas[idx_atual]:.4f}"
            )

    indices_mantidos = sorted(melhor_por_classe.values())

    return (
        boxes[indices_mantidos],
        masks[indices_mantidos],
        [nome_classes[i] for i in indices_mantidos],
        confiancas[indices_mantidos],
        duplicadas
    )


def validar_pecas_detectadas(montagem):
    classes_detectadas = set(montagem["classes"])
    ausentes = CLASSES_ESPERADAS - classes_detectadas
    extras = classes_detectadas - CLASSES_ESPERADAS

    if not ausentes and not extras and len(montagem["classes"]) == len(CLASSES_ESPERADAS):
        return True, ""

    mensagens = []
    if ausentes:
        mensagens.append(f"peças ausentes: {sorted(ausentes)}")
    if extras:
        mensagens.append(f"classes inesperadas: {sorted(extras)}")

    return False, " | ".join(mensagens)


def calcular_distancia(c1, c2):
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def calcular_angulo(c1, c2):
    dx, dy = c2[0] - c1[0], c2[1] - c1[1]
    ang = np.degrees(np.arctan2(dy, dx))
    return ang + 360 if ang < 0 else ang


def normalizar_angulo(angulo):
    return angulo % 360


def normalizar_eixo(angulo):
    return angulo % 180


def diferenca_angular(a1, a2):
    diff = abs(a1 - a2)
    return min(diff, 360 - diff)


def diferenca_eixo(a1, a2):
    diff = abs(a1 - a2)
    return min(diff, 180 - diff)


def calcular_rotacao_global(gabarito_atual, montagem_atual):
    idx_gabarito = {classe: i for i, classe in enumerate(gabarito_atual["classes"])}
    idx_montagem = {classe: i for i, classe in enumerate(montagem_atual["classes"])}

    def vetor_principal(dados, idx):
        c_esq = dados["centroides_norm"][idx["Esquerda"]]
        c_cima = dados["centroides_norm"][idx["Direita_cima"]]
        c_baixo = dados["centroides_norm"][idx["Direita_baixo"]]
        centro_direita = (c_cima + c_baixo) / 2
        return calcular_angulo(c_esq, centro_direita)

    ang_ref = vetor_principal(gabarito_atual, idx_gabarito)
    ang_novo = vetor_principal(montagem_atual, idx_montagem)
    rotacao = normalizar_angulo(ang_novo - ang_ref)

    if rotacao > 180:
        rotacao -= 360

    return rotacao


def calcular_contato(mascara_a, mascara_b):
    kernel = np.ones((11, 11), np.uint8)
    mascara_a = mascara_a.astype(np.uint8)
    mascara_b = mascara_b.astype(np.uint8)

    contato_a = np.logical_and(cv2.dilate(mascara_a, kernel) > 0, mascara_b).sum()
    contato_b = np.logical_and(cv2.dilate(mascara_b, kernel) > 0, mascara_a).sum()
    contato_total = contato_a + contato_b

    borda_a = np.logical_and(mascara_a, cv2.erode(mascara_a, kernel) == 0).sum()
    borda_b = np.logical_and(mascara_b, cv2.erode(mascara_b, kernel) == 0).sum()

    return contato_total / max(1, min(borda_a, borda_b))


def avaliar_par(nome_par, gabarito_atual, montagem_atual, rotacao_global):
    pares_classes = {
        "esquerda_cima": ("Esquerda", "Direita_cima"),
        "esquerda_baixo": ("Esquerda", "Direita_baixo"),
        "cima_baixo": ("Direita_cima", "Direita_baixo"),
    }

    idx_gabarito = {classe: i for i, classe in enumerate(gabarito_atual["classes"])}
    idx_montagem = {classe: i for i, classe in enumerate(montagem_atual["classes"])}
    classe_a, classe_b = pares_classes[nome_par]

    idx_a_ref = idx_gabarito[classe_a]
    idx_b_ref = idx_gabarito[classe_b]
    idx_a_novo = idx_montagem[classe_a]
    idx_b_novo = idx_montagem[classe_b]
    
    c_a_ref = gabarito_atual["centroides_norm"][idx_a_ref]
    c_b_ref = gabarito_atual["centroides_norm"][idx_b_ref]
    c_a_novo = montagem_atual["centroides_norm"][idx_a_novo]
    c_b_novo = montagem_atual["centroides_norm"][idx_b_novo]
    
    ang_a_ref = gabarito_atual["angulos_abs"][idx_a_ref]
    ang_b_ref = gabarito_atual["angulos_abs"][idx_b_ref]
    ang_a_novo = montagem_atual["angulos_abs"][idx_a_novo]
    ang_b_novo = montagem_atual["angulos_abs"][idx_b_novo]
    mask_a_novo = montagem_atual["masks_img"][idx_a_novo]
    mask_b_novo = montagem_atual["masks_img"][idx_b_novo]
    
    # Critério 1: Distância
    dist_ref = calcular_distancia(c_a_ref, c_b_ref)
    dist_novo = calcular_distancia(c_a_novo, c_b_novo)
    erro_dist = abs(dist_novo - dist_ref) / dist_ref
    crit1_ok = erro_dist <= TOL_DIST_FRAC
    
    # Critério 2: Ângulo relativo
    ang_rel_ref = calcular_angulo(c_a_ref, c_b_ref)
    ang_rel_novo = calcular_angulo(c_a_novo, c_b_novo)
    ang_rel_ref_compensado = normalizar_angulo(ang_rel_ref + rotacao_global)
    erro_ang = diferenca_angular(ang_rel_novo, ang_rel_ref_compensado)
    crit2_ok = erro_ang <= TOL_ANG_DEG
    
    # Critério 3: Rotação da peça, compensando a rotação global da foto
    ang_a_ref_compensado = normalizar_eixo(ang_a_ref + rotacao_global)
    ang_b_ref_compensado = normalizar_eixo(ang_b_ref + rotacao_global)
    erro_rot_a = diferenca_eixo(ang_a_novo, ang_a_ref_compensado)
    erro_rot_b = diferenca_eixo(ang_b_novo, ang_b_ref_compensado)
    crit3_ok = (erro_rot_a <= TOL_ROT_DEG) and (erro_rot_b <= TOL_ROT_DEG)

    # Critério 4: contato real entre as máscaras da junção
    contato = calcular_contato(mask_a_novo, mask_b_novo)
    crit4_ok = contato >= TOL_CONTATO_MIN
    
    return {
        "par": nome_par,
        "crit1": crit1_ok,
        "crit1_valor": erro_dist,
        "crit2": crit2_ok,
        "crit2_valor": erro_ang,
        "crit3": crit3_ok,
        "crit3_valor_a": erro_rot_a,
        "crit3_valor_b": erro_rot_b,
        "crit4": crit4_ok,
        "crit4_valor": contato,
        "passou": crit1_ok and crit2_ok and crit3_ok and crit4_ok
    }


def avaliar_montagem(img_path, gabarito_path=GABARITO_PATH):
    """Avalia montagem completa."""
    img_path = str(img_path)
    
    print("\n" + "="*60)
    print(f"AVALIANDO MONTAGEM: {img_path}")
    print("="*60)
    
    print("\n=== ETAPA 1: CARREGANDO GABARITO ===")
    gabarito = carregar_gabarito(gabarito_path)
    print(f"✅ Gabarito carregado: {gabarito['classes']}")
    
    print("\n=== ETAPA 2: SEGMENTANDO IMAGEM ===")
    montagem = segmentar_imagem(img_path, gabarito)
    print(f"✅ Imagem segmentada: {montagem['classes']}")

    deteccao_ok, motivo = validar_pecas_detectadas(montagem)
    if not deteccao_ok:
        print("\n❌ Avaliação da montagem não foi possível.")
        print(f"   Nem todas as peças foram detectadas corretamente: {motivo}")
        print("\n" + "="*60)
        print("NOTA FINAL: 0/3")
        print("="*60 + "\n")
        return {
            "nota": 0,
            "detalhes": [],
            "arquivo": img_path,
            "classes_detectadas": montagem["classes"],
            "duplicadas": montagem["duplicadas"],
            "juncoes_falhas": [],
            "rotacao_global": "",
            "erro": motivo
        }
    
    print("\n=== ETAPA 3: AVALIANDO PARES ===")
    pares = ["esquerda_cima", "esquerda_baixo", "cima_baixo"]
    resultados = []
    nota = 0
    rotacao_global = calcular_rotacao_global(gabarito, montagem)
    print(f"Rotação global estimada da foto: {rotacao_global:.1f}°")
    
    for par in pares:
        resultado = avaliar_par(par, gabarito, montagem, rotacao_global)
        resultados.append(resultado)
        
        status = "✅" if resultado["passou"] else "❌"
        print(f"\n{status} {par.upper()}")
        print(f"   Crit1 (Dist): {resultado['crit1']} (erro={resultado['crit1_valor']:.2%})")
        print(f"   Crit2 (Ang):  {resultado['crit2']} (erro={resultado['crit2_valor']:.1f}°)")
        print(f"   Crit3 (Rot):  {resultado['crit3']} (erro_a={resultado['crit3_valor_a']:.1f}°, erro_b={resultado['crit3_valor_b']:.1f}°)")
        print(f"   Crit4 (Contato): {resultado['crit4']} (valor={resultado['crit4_valor']:.2f})")
        
        if resultado["passou"]:
            nota += 1

    juncoes_falhas = [resultado["par"] for resultado in resultados if not resultado["passou"]]
    
    print("\n" + "="*60)
    print(f"NOTA FINAL: {nota}/3")
    print("="*60 + "\n")
    
    return {
        "nota": nota,
        "detalhes": resultados,
        "arquivo": img_path,
        "classes_detectadas": montagem["classes"],
        "duplicadas": montagem["duplicadas"],
        "juncoes_falhas": juncoes_falhas,
        "rotacao_global": rotacao_global,
        "erro": ""
    }


def listar_imagens(pasta):
    pasta = Path(pasta)
    imagens = [
        caminho
        for caminho in pasta.iterdir()
        if caminho.is_file() and caminho.suffix.lower() in EXTENSOES_IMAGEM
    ]
    return sorted(imagens, key=lambda caminho: ordem_natural(caminho.name))


def ordem_natural(texto):
    partes = re.split(r"(\d+)", texto.lower())
    return [int(parte) if parte.isdigit() else parte for parte in partes]


def formatar_observacoes(resultado):
    observacoes = []

    if resultado["duplicadas"]:
        observacoes.append("Duplicadas removidas: " + "; ".join(resultado["duplicadas"]))

    if resultado["erro"]:
        observacoes.append("Avaliação não possível: " + resultado["erro"])

    if resultado["juncoes_falhas"]:
        observacoes.append("Junções que falharam: " + ", ".join(resultado["juncoes_falhas"]))

    if not observacoes:
        observacoes.append("Sem observações")

    return " | ".join(observacoes)


def salvar_resultados_excel(resultados, excel_output=EXCEL_OUTPUT):
    wb = Workbook()
    ws = wb.active
    ws.title = "Resultados"

    ws.append([
        "Imagem",
        "Nota",
        "Status",
        "Rotação global estimada",
        "Classes detectadas",
        "Junções que falharam",
        "Observações"
    ])

    for resultado in resultados:
        nome_imagem = Path(resultado["arquivo"]).name
        nota = resultado["nota"]

        if resultado["erro"]:
            status = "ERRO DETECÇÃO"
        elif nota == 3:
            status = "CORRETA"
        elif nota > 0:
            status = "PARCIAL"
        else:
            status = "ERRADA"

        ws.append([
            nome_imagem,
            nota,
            status,
            resultado["rotacao_global"],
            ", ".join(resultado["classes_detectadas"]),
            ", ".join(resultado["juncoes_falhas"]),
            formatar_observacoes(resultado)
        ])

    wb.save(excel_output)
    print(f"\nResultados salvos em: {excel_output}")


def avaliar_pasta(pasta=IMAGENS_AVALIACAO_DIR, gabarito_path=GABARITO_PATH, excel_output=EXCEL_OUTPUT):
    imagens = listar_imagens(pasta)

    if not imagens:
        print(f"Nenhuma imagem encontrada em: {pasta}")
        return []

    resultados = []
    for imagem in imagens:
        resultados.append(avaliar_montagem(imagem, gabarito_path))

    salvar_resultados_excel(resultados, excel_output)
    return resultados


if __name__ == "__main__":
    avaliar_pasta()
