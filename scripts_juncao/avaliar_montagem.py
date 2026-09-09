import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path
from openpyxl import Workbook
import re
import json
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG = False
EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolver_caminho(caminho):
    caminho = Path(caminho)
    if caminho.is_absolute():
        return caminho
    return PROJECT_ROOT / caminho


def caminho_config_figura(nome_figura):
    return PROJECT_ROOT / "figuras" / nome_figura / "config.json"


def carregar_config(config_path):
    config_path = resolver_caminho(config_path)
    with config_path.open("r", encoding="utf-8") as arquivo:
        config = json.load(arquivo)

    config["config_path"] = str(config_path)
    config["modelo"] = resolver_caminho(config["modelo"])
    config["gabarito"] = resolver_caminho(config["gabarito"])
    config["pasta_avaliacao"] = resolver_caminho(config["pasta_avaliacao"])
    config["saida_excel"] = resolver_caminho(config["saida_excel"])
    config["classes_esperadas"] = set(config["classes_esperadas"])
    config["pares"] = {
        nome_par: tuple(classes)
        for nome_par, classes in config["pares"].items()
    }
    return config


def carregar_modelo(config):
    return YOLO(str(config["modelo"]))


def carregar_gabarito(npz_path):
    dados = np.load(npz_path, allow_pickle=True)

    campos_obrigatorios = {
        "schema_version",
        "figura",
        "classes",
        "centroides_norm",
        "angulos_abs",
        "pares_nomes",
        "pares_classes",
        "pares_distancias",
        "pares_angulos"
    }
    campos_ausentes = campos_obrigatorios - set(dados.files)
    if campos_ausentes:
        raise ValueError(
            f"Gabarito em formato incompatível. Campos ausentes: {sorted(campos_ausentes)}"
        )

    pares_nomes = list(dados["pares_nomes"])
    pares_classes = dados["pares_classes"]
    pares_distancias = dados["pares_distancias"]
    pares_angulos = dados["pares_angulos"]

    relacoes_pares = {}
    for i, nome_par in enumerate(pares_nomes):
        relacoes_pares[str(nome_par)] = {
            "classes": tuple(str(classe) for classe in pares_classes[i]),
            "dist": float(pares_distancias[i]),
            "ang": float(pares_angulos[i])
        }

    return {
        "schema_version": int(dados["schema_version"]),
        "figura": str(dados["figura"]),
        "classes": [str(classe) for classe in dados["classes"]],
        "centroides_norm": dados["centroides_norm"],
        "angulos_abs": dados["angulos_abs"],
        "relacoes_pares": relacoes_pares,
        "contatos_norm": {
            str(nome_par): dados["pares_contato_centroides"][i]
            for i, nome_par in enumerate(pares_nomes)
        } if "pares_contato_centroides" in dados.files else {},
    }


def segmentar_imagem(img_path, modelo, conf_min):
    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {img_path}")

    resultado = modelo(img, conf=conf_min, verbose=DEBUG)[0]

    if resultado.masks is None or len(resultado.boxes) == 0:
        return {
            "classes": [],
            "centroides_norm": np.array([]),
            "angulos_abs": np.array([]),
            "confiancas": np.array([]),
            "masks_img": np.array([]),
            "limites_figura": np.array([]),
            "duplicadas": []
        }
    
    boxes = resultado.boxes.xyxy.cpu().numpy()
    masks = resultado.masks.data.cpu().numpy()
    classes = resultado.boxes.cls.cpu().numpy()
    confiancas = resultado.boxes.conf.cpu().numpy()

    nome_classes = [modelo.names[int(c)] for c in classes]
    boxes, masks, nome_classes, confiancas, duplicadas = deduplicar_por_confianca(
        boxes,
        masks,
        nome_classes,
        confiancas
    )

    if DEBUG and duplicadas:
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
        "limites_figura": np.array([x_min, y_min, x_max, y_max]),
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


def validar_pecas_detectadas(montagem, classes_esperadas):
    classes_detectadas = set(montagem["classes"])
    ausentes = classes_esperadas - classes_detectadas
    extras = classes_detectadas - classes_esperadas

    if not ausentes and not extras and len(montagem["classes"]) == len(classes_esperadas):
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


def calcular_rotacao_global(gabarito_atual, montagem_atual, config):
    idx_gabarito = {classe: i for i, classe in enumerate(gabarito_atual["classes"])}
    idx_montagem = {classe: i for i, classe in enumerate(montagem_atual["classes"])}
    rotacao_config = config["rotacao_global"]

    if rotacao_config.get("metodo") == "mediana_angulos_pecas":
        # As peças podem estar deslocadas em uma montagem incorreta. Nesse
        # caso, estimar a rotação pela geometria da montagem inteira torna a
        # referência instável. A mediana das orientações é robusta: uma peça
        # fora de posição não domina a rotação global estimada.
        rotacoes = []
        for classe in gabarito_atual["classes"]:
            ang_ref = gabarito_atual["angulos_abs"][idx_gabarito[classe]]
            ang_novo = montagem_atual["angulos_abs"][idx_montagem[classe]]
            diferenca = (ang_novo - ang_ref) % 180
            if diferenca > 90:
                diferenca -= 180
            rotacoes.append(diferenca)
        return float(np.median(rotacoes))

    def vetor_principal(dados, idx):
        origem = dados["centroides_norm"][idx[rotacao_config["origem"]]]
        destino_classes = rotacao_config["destino_centroide"]
        destino = np.mean(
            [dados["centroides_norm"][idx[classe]] for classe in destino_classes],
            axis=0
        )
        return calcular_angulo(origem, destino)

    ang_ref = vetor_principal(gabarito_atual, idx_gabarito)
    ang_novo = vetor_principal(montagem_atual, idx_montagem)
    rotacao = normalizar_angulo(ang_novo - ang_ref)

    if rotacao > 180:
        rotacao -= 360

    return rotacao


def calcular_contato_e_centroide(mascara_a, mascara_b):
    kernel = np.ones((11, 11), np.uint8)
    mascara_a = mascara_a.astype(np.uint8)
    mascara_b = mascara_b.astype(np.uint8)

    contato_a = np.logical_and(cv2.dilate(mascara_a, kernel) > 0, mascara_b).sum()
    contato_b = np.logical_and(cv2.dilate(mascara_b, kernel) > 0, mascara_a).sum()
    contato_total = contato_a + contato_b

    borda_a = np.logical_and(mascara_a, cv2.erode(mascara_a, kernel) == 0).sum()
    borda_b = np.logical_and(mascara_b, cv2.erode(mascara_b, kernel) == 0).sum()

    regiao_contato = np.logical_or(
        np.logical_and(cv2.dilate(mascara_a, kernel) > 0, mascara_b),
        np.logical_and(cv2.dilate(mascara_b, kernel) > 0, mascara_a),
    )
    y_pixels, x_pixels = np.where(regiao_contato)
    centroide = None if not len(x_pixels) else np.array([np.mean(x_pixels), np.mean(y_pixels)])

    return contato_total / max(1, min(borda_a, borda_b)), centroide


def coordenadas_locais_do_contato(centroide_contato, centroide_a, centroide_b):
    """Expressa o contato no referencial formado pelo próprio par de peças.

    O primeiro valor indica a posição ao longo do vetor A -> B; o segundo,
    a posição lateral. Assim, a medida não depende de onde o caminhão está na
    foto, nem da rotação global da fotografia.
    """
    vetor_ab = centroide_b - centroide_a
    norma_quadrada = float(np.dot(vetor_ab, vetor_ab))
    if norma_quadrada == 0:
        return None

    vetor_contato = centroide_contato - centroide_a
    return np.array([
        np.dot(vetor_contato, vetor_ab) / norma_quadrada,
        (vetor_ab[0] * vetor_contato[1] - vetor_ab[1] * vetor_contato[0])
        / norma_quadrada,
    ])


def avaliar_par(nome_par, gabarito_atual, montagem_atual, rotacao_global, config):
    pares_classes = config["pares"]
    tolerancias = config["tolerancias"]
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
    crit1_ok = erro_dist <= tolerancias["distancia_frac"]
    
    # Critério 2: Ângulo relativo
    ang_rel_ref = calcular_angulo(c_a_ref, c_b_ref)
    ang_rel_novo = calcular_angulo(c_a_novo, c_b_novo)
    ang_rel_ref_compensado = normalizar_angulo(ang_rel_ref + rotacao_global)
    erro_ang = diferenca_angular(ang_rel_novo, ang_rel_ref_compensado)
    crit2_ok = erro_ang <= tolerancias["angulo_graus"]
    
    # Critério 3: Rotação da peça, compensando a rotação global da foto
    ang_a_ref_compensado = normalizar_eixo(ang_a_ref + rotacao_global)
    ang_b_ref_compensado = normalizar_eixo(ang_b_ref + rotacao_global)
    erro_rot_a = diferenca_eixo(ang_a_novo, ang_a_ref_compensado)
    erro_rot_b = diferenca_eixo(ang_b_novo, ang_b_ref_compensado)
    crit3_ok = (
        erro_rot_a <= tolerancias["rotacao_graus"]
        and erro_rot_b <= tolerancias["rotacao_graus"]
    )

    # Critério 4: contato real entre as máscaras da junção
    contato, centroide_contato = calcular_contato_e_centroide(mask_a_novo, mask_b_novo)
    contato_minimos = tolerancias.get("contato_min_por_par", {})
    contato_minimo = contato_minimos.get(nome_par, tolerancias["contato_min"])
    crit4_ok = contato >= contato_minimo

    # Critério 5 (opcional): o contato ocorreu na região esperada da junção.
    # É configurado por par, portanto figuras que não o habilitam (como a
    # maçã) preservam exatamente os critérios anteriores.
    crit5_ok = True
    erro_contato_posicao = None
    tolerancias_posicao = tolerancias.get("contato_posicao_local_por_par", {})
    tolerancia_posicao = tolerancias_posicao.get(nome_par)
    contato_ref = gabarito_atual.get("contatos_norm", {}).get(nome_par)
    if tolerancia_posicao is not None and contato_ref is not None:
        if centroide_contato is None:
            crit5_ok = False
            erro_contato_posicao = float("inf")
        else:
            x_min, y_min, x_max, y_max = montagem_atual["limites_figura"]
            centroide_norm = np.array([
                (centroide_contato[0] - x_min) / (x_max - x_min),
                (centroide_contato[1] - y_min) / (y_max - y_min),
            ])
            contato_ref_local = coordenadas_locais_do_contato(
                contato_ref, c_a_ref, c_b_ref
            )
            contato_novo_local = coordenadas_locais_do_contato(
                centroide_norm, c_a_novo, c_b_novo
            )
            if contato_ref_local is None or contato_novo_local is None:
                crit5_ok = False
                erro_contato_posicao = float("inf")
            else:
                erro_contato_posicao = float(
                    np.linalg.norm(contato_novo_local - contato_ref_local)
                )
                crit5_ok = erro_contato_posicao <= tolerancia_posicao
    
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
        "crit4_minimo": contato_minimo,
        "crit5": crit5_ok,
        "crit5_valor": erro_contato_posicao,
        "crit5_tolerancia": tolerancia_posicao,
        "passou": crit1_ok and crit2_ok and crit3_ok and crit4_ok and crit5_ok
    }


def avaliar_montagem(img_path, config, modelo):
    """Avalia montagem completa."""
    img_path = str(img_path)
    total_pares = len(config["pares"])
    
    print("\n" + "="*60)
    print(f"AVALIANDO MONTAGEM: {img_path}")
    print("="*60)
    
    gabarito = carregar_gabarito(config["gabarito"])
    if DEBUG:
        print("\n=== ETAPA 1: CARREGANDO GABARITO ===")
        print(f"✅ Gabarito carregado: {gabarito['classes']}")
    
    if DEBUG:
        print("\n=== ETAPA 2: SEGMENTANDO IMAGEM ===")
    montagem = segmentar_imagem(img_path, modelo, config["confianca_avaliacao"])
    if DEBUG:
        print(f"✅ Imagem segmentada: {montagem['classes']}")

    deteccao_ok, motivo = validar_pecas_detectadas(montagem, config["classes_esperadas"])
    if not deteccao_ok:
        print("\n❌ Avaliação da montagem não foi possível.")
        print(f"Motivo: {motivo}")
        print("\n" + "="*60)
        print(f"NOTA FINAL: 0/{total_pares}")
        print("="*60 + "\n")
        return {
            "nota": 0,
            "nota_maxima": total_pares,
            "detalhes": [],
            "arquivo": img_path,
            "classes_detectadas": montagem["classes"],
            "duplicadas": montagem["duplicadas"],
            "juncoes_falhas": [],
            "rotacao_global": "",
            "erro": motivo
        }
    
    pares = list(config["pares"])
    resultados = []
    nota = 0
    rotacao_global = calcular_rotacao_global(gabarito, montagem, config)

    if DEBUG:
        print("\n=== ETAPA 3: AVALIANDO PARES ===")
        print(f"Rotação global estimada da foto: {rotacao_global:.1f}°")
    
    for par in pares:
        resultado = avaliar_par(par, gabarito, montagem, rotacao_global, config)
        resultados.append(resultado)
        
        if DEBUG:
            status = "✅" if resultado["passou"] else "❌"
            print(f"\n{status} {par.upper()}")
            print(f"   Crit1 (Dist): {resultado['crit1']} (erro={resultado['crit1_valor']:.2%})")
            print(f"   Crit2 (Ang):  {resultado['crit2']} (erro={resultado['crit2_valor']:.1f}°)")
            print(f"   Crit3 (Rot):  {resultado['crit3']} (erro_a={resultado['crit3_valor_a']:.1f}°, erro_b={resultado['crit3_valor_b']:.1f}°)")
            print(
                f"   Crit4 (Contato): {resultado['crit4']} "
                f"(valor={resultado['crit4_valor']:.2f}, "
                f"mínimo={resultado['crit4_minimo']:.2f})"
            )
            if resultado["crit5_tolerancia"] is not None:
                print(
                    f"   Crit5 (Posição do contato): {resultado['crit5']} "
                    f"(erro={resultado['crit5_valor']:.2f}, "
                    f"máximo={resultado['crit5_tolerancia']:.2f})"
                )
        
        if resultado["passou"]:
            nota += 1

    juncoes_falhas = [resultado["par"] for resultado in resultados if not resultado["passou"]]
    status_final = obter_status(nota, erro="", nota_maxima=total_pares)
    
    print("\n" + "="*60)
    print(f"NOTA FINAL: {nota}/{total_pares}")
    print(f"STATUS: {status_final}")
    if juncoes_falhas:
        print(f"JUNÇÕES QUE PRECISAM DE ATENÇÃO: {', '.join(juncoes_falhas)}")
    else:
        print("MONTAGEM SEM JUNÇÕES REPROVADAS")
    print("="*60 + "\n")
    
    return {
        "nota": nota,
        "nota_maxima": total_pares,
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


def obter_status(nota, erro, nota_maxima):
    if erro:
        return "ERRO DETECÇÃO"
    if nota == nota_maxima:
        return "CORRETA"
    if nota > 0:
        return "PARCIAL"
    return "ERRADA"


def formatar_detalhes_tecnicos(resultado):
    linhas = []
    for detalhe in resultado["detalhes"]:
        linhas.append(
            f"{detalhe['par']}: "
            f"Crit1 Dist={detalhe['crit1']} erro={detalhe['crit1_valor']:.2%}; "
            f"Crit2 Ang={detalhe['crit2']} erro={detalhe['crit2_valor']:.1f}°; "
            f"Crit3 Rot={detalhe['crit3']} erro_a={detalhe['crit3_valor_a']:.1f}° erro_b={detalhe['crit3_valor_b']:.1f}°; "
            f"Crit4 Contato={detalhe['crit4']} valor={detalhe['crit4_valor']:.2f}"
        )
    return " | ".join(linhas)


def salvar_resultados_excel(resultados, excel_output):
    wb = Workbook()
    ws = wb.active
    ws.title = "Resultados"

    cabecalho = [
        "Imagem",
        "Nota",
        "Status",
        "Classes detectadas",
        "Junções que falharam",
        "Observações"
    ]

    if DEBUG:
        cabecalho.insert(3, "Rotação global estimada")
        cabecalho.append("Detalhes técnicos")

    ws.append(cabecalho)

    for resultado in resultados:
        nome_imagem = Path(resultado["arquivo"]).name
        nota = resultado["nota"]
        status = obter_status(nota, resultado["erro"], resultado["nota_maxima"])

        linha = [
            nome_imagem,
            f"{nota}/{resultado['nota_maxima']}",
            status,
            ", ".join(resultado["classes_detectadas"]),
            ", ".join(resultado["juncoes_falhas"]),
            formatar_observacoes(resultado)
        ]

        if DEBUG:
            linha.insert(3, resultado["rotacao_global"])
            linha.append(formatar_detalhes_tecnicos(resultado))

        ws.append(linha)

    wb.save(excel_output)
    print(f"\nResultados salvos em: {excel_output}")


def avaliar_pasta(config_path):
    config = carregar_config(config_path)
    modelo = carregar_modelo(config)
    imagens = listar_imagens(config["pasta_avaliacao"])

    if not imagens:
        print(f"Nenhuma imagem encontrada em: {config['pasta_avaliacao']}")
        return []

    resultados = []
    for imagem in imagens:
        resultados.append(avaliar_montagem(imagem, config, modelo))

    salvar_resultados_excel(resultados, config["saida_excel"])
    return resultados


def parse_args():
    parser = argparse.ArgumentParser(
        description="Avalia montagens de uma figura configurada."
    )
    parser.add_argument(
        "--figura",
        required=True,
        help="Nome da figura em figuras/<nome>/config.json. Exemplo: maca."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    avaliar_pasta(caminho_config_figura(args.figura))
