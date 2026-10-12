import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    config["imagem_gabarito"] = resolver_caminho(config["imagem_gabarito"])
    config["gabarito"] = resolver_caminho(config["gabarito"])
    config["classes_esperadas"] = set(config["classes_esperadas"])
    config["pares"] = {
        nome_par: tuple(classes)
        for nome_par, classes in config["pares"].items()
    }
    return config


def carregar_modelo(config):
    return YOLO(str(config["modelo"]))


def processar_imagem(img_path, modelo, conf_min) -> dict:
    img = cv2.imread(str(img_path))

    if img is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {img_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultado = modelo(img, conf=conf_min)[0]

    if resultado.masks is None or len(resultado.boxes) == 0:
        raise ValueError(f"Nenhuma peça foi detectada na imagem de gabarito: {img_path}")

    boxes = resultado.boxes.xyxy.cpu().numpy()
    masks = resultado.masks.data.cpu().numpy()
    classes = resultado.boxes.cls.cpu().numpy()
    confiancas = resultado.boxes.conf.cpu().numpy()
    nome_classes = [modelo.names[int(c)] for c in classes]

    return {
        "imagem": img_rgb,
        "boxes": boxes,
        "masks": masks,
        "classes": nome_classes,
        "confiancas": confiancas
    }


def validar_pecas_detectadas(classes_detectadas, classes_esperadas):
    detectadas = set(classes_detectadas)
    ausentes = classes_esperadas - detectadas
    extras = detectadas - classes_esperadas

    if not ausentes and not extras and len(classes_detectadas) == len(classes_esperadas):
        return

    mensagens = []
    if ausentes:
        mensagens.append(f"peças ausentes: {sorted(ausentes)}")
    if extras:
        mensagens.append(f"classes inesperadas: {sorted(extras)}")
    if len(classes_detectadas) != len(detectadas):
        mensagens.append("detecções duplicadas")

    raise ValueError("Gabarito inválido: " + " | ".join(mensagens))


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

    for mascara in masks:
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


def calcular_angulo_orientacao(masks, imagem_shape) -> np.ndarray:
    angulos = []
    altura_img, largura_img = imagem_shape[:2]
    altura_mask, largura_mask = masks.shape[1:3]
    escala_x = largura_img / largura_mask
    escala_y = altura_img / altura_mask

    for mascara in masks:
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


def calcular_angulo(c1, c2):
    dx, dy = c2[0] - c1[0], c2[1] - c1[1]
    ang = np.degrees(np.arctan2(dy, dx))
    return ang + 360 if ang < 0 else ang


def calcular_relacoes_pares(centroides_norm, classes, pares) -> dict:
    indice_classes = {classe: i for i, classe in enumerate(classes)}
    relacoes = {}

    for nome_par, (classe_a, classe_b) in pares.items():
        if classe_a not in indice_classes or classe_b not in indice_classes:
            raise ValueError(
                f"Par '{nome_par}' referencia classes não detectadas: "
                f"{classe_a}, {classe_b}"
            )

        centroide_a = centroides_norm[indice_classes[classe_a]]
        centroide_b = centroides_norm[indice_classes[classe_b]]
        distancia = np.sqrt(
            (centroide_a[0] - centroide_b[0])**2
            + (centroide_a[1] - centroide_b[1])**2
        )
        angulo = calcular_angulo(centroide_a, centroide_b)

        relacoes[nome_par] = {
            "classes": (classe_a, classe_b),
            "distancia": float(distancia),
            "angulo": float(angulo)
        }

    return relacoes


def calcular_contatos_normalizados(masks, boxes, classes, imagem_shape, pares) -> dict:
    """Calcula o centro da região de contato de cada junção do gabarito."""
    x_min, y_min = np.min(boxes[:, 0]), np.min(boxes[:, 1])
    x_max, y_max = np.max(boxes[:, 2]), np.max(boxes[:, 3])
    largura_total, altura_total = x_max - x_min, y_max - y_min
    altura_img, largura_img = imagem_shape[:2]

    mascaras_img = [
        cv2.resize(mascara.astype(np.uint8), (largura_img, altura_img), interpolation=cv2.INTER_NEAREST)
        for mascara in masks
    ]
    indice_classes = {classe: i for i, classe in enumerate(classes)}
    kernel = np.ones((11, 11), np.uint8)
    contatos = {}

    for nome_par, (classe_a, classe_b) in pares.items():
        mascara_a = mascaras_img[indice_classes[classe_a]]
        mascara_b = mascaras_img[indice_classes[classe_b]]
        regiao_a = np.logical_and(cv2.dilate(mascara_a, kernel) > 0, mascara_b)
        regiao_b = np.logical_and(cv2.dilate(mascara_b, kernel) > 0, mascara_a)
        regiao = np.logical_or(regiao_a, regiao_b)
        y_pixels, x_pixels = np.where(regiao)

        if not len(x_pixels):
            raise ValueError(f"Não foi possível localizar o contato da junção '{nome_par}'.")

        contatos[nome_par] = np.array([
            (np.mean(x_pixels) - x_min) / largura_total,
            (np.mean(y_pixels) - y_min) / altura_total,
        ])

    return contatos


def salvar_gabarito(
    centroides_norm,
    angulos_abs,
    relacoes_pares,
    classes,
    config,
    contatos_norm=None,
) -> None:
    npz_path = config["gabarito"]
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    pares_nomes = []
    pares_classes = []
    pares_distancias = []
    pares_angulos = []

    for nome_par, dados in relacoes_pares.items():
        pares_nomes.append(nome_par)
        pares_classes.append(dados["classes"])
        pares_distancias.append(dados["distancia"])
        pares_angulos.append(dados["angulo"])

    dados = {
        "schema_version": np.array(3 if contatos_norm is not None else 2),
        "figura": np.array(config["nome"]),
        "classes": np.array(classes),
        "centroides_norm": centroides_norm,
        "angulos_abs": angulos_abs,
        "pares_nomes": np.array(pares_nomes),
        "pares_classes": np.array(pares_classes),
        "pares_distancias": np.array(pares_distancias),
        "pares_angulos": np.array(pares_angulos),
    }
    if contatos_norm is not None:
        dados["pares_contato_centroides"] = np.array([
            contatos_norm[nome_par] for nome_par in pares_nomes
        ])

    np.savez_compressed(
        npz_path,
        **dados,
    )

    print(f"Gabarito salvo em: {npz_path}")


def carregar_gabarito(npz_path) -> dict:
    dados = np.load(npz_path, allow_pickle=True)

    pares_nomes = list(dados["pares_nomes"])
    pares_classes = dados["pares_classes"]
    pares_distancias = dados["pares_distancias"]
    pares_angulos = dados["pares_angulos"]

    relacoes_pares = {}
    for i, nome_par in enumerate(pares_nomes):
        relacoes_pares[str(nome_par)] = {
            "classes": tuple(str(classe) for classe in pares_classes[i]),
            "distancia": float(pares_distancias[i]),
            "angulo": float(pares_angulos[i])
        }

    contatos_norm = {}
    if "pares_contato_centroides" in dados.files:
        for i, nome_par in enumerate(pares_nomes):
            contatos_norm[str(nome_par)] = dados["pares_contato_centroides"][i]

    return {
        "schema_version": int(dados["schema_version"]),
        "figura": str(dados["figura"]),
        "classes": [str(classe) for classe in dados["classes"]],
        "centroides_norm": dados["centroides_norm"],
        "angulos_abs": dados["angulos_abs"],
        "relacoes_pares": relacoes_pares,
        "contatos_norm": contatos_norm,
    }


def gerar_gabarito(config_path):
    config = carregar_config(config_path)
    modelo = carregar_modelo(config)

    print("=" * 60)
    print(f"GERANDO GABARITO DE SEGMENTAÇÃO: {config['nome']}")
    print("=" * 60)

    print("\n=== ETAPA 1: SEGMENTAÇÃO ===")
    resultado = processar_imagem(
        config["imagem_gabarito"],
        modelo,
        config["confianca_gabarito"]
    )
    validar_pecas_detectadas(resultado["classes"], config["classes_esperadas"])
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

    print("\n=== ETAPA 2B: ÂNGULOS DE ORIENTAÇÃO ===")
    angulos = calcular_angulo_orientacao(
        resultado["masks"],
        resultado["imagem"].shape
    )
    print("Ângulos de Orientação:")
    for i, classe in enumerate(resultado["classes"]):
        print(f"  {classe}: {angulos[i]:.2f}°")

    print("\n=== ETAPA 3: RELAÇÕES ENTRE PARES ===")
    relacoes = calcular_relacoes_pares(
        centroides_norm,
        resultado["classes"],
        config["pares"]
    )

    contatos_norm = None
    if "contato_posicao_local_por_par" in config["tolerancias"]:
        contatos_norm = calcular_contatos_normalizados(
            resultado["masks"],
            resultado["boxes"],
            resultado["classes"],
            resultado["imagem"].shape,
            config["pares"],
        )
    print("Relações entre Pares:")
    for par, dados in relacoes.items():
        print(f"  {par}:")
        print(f"    - Classes: {dados['classes'][0]} x {dados['classes'][1]}")
        print(f"    - Distância: {dados['distancia']:.4f}")
        print(f"    - Ângulo Relativo: {dados['angulo']:.2f}°")

    print("\n=== ETAPA 4: PERSISTÊNCIA ===")
    salvar_gabarito(
        centroides_norm,
        angulos,
        relacoes,
        resultado["classes"],
        config,
        contatos_norm,
    )

    print("\n=== VERIFICAÇÃO: Carregando Gabarito ===")
    gabarito = carregar_gabarito(config["gabarito"])
    print("Gabarito carregado com sucesso!")
    print(f"Schema: {gabarito['schema_version']}")
    print(f"Figura: {gabarito['figura']}")
    print(f"Classes: {gabarito['classes']}")
    print(f"Centróides normalizados:\n{gabarito['centroides_norm']}")
    print(f"Ângulos absolutos:\n{gabarito['angulos_abs']}")
    print("Relações entre pares:")
    for par, dados in gabarito["relacoes_pares"].items():
        print(
            f"  {par}: classes={dados['classes']} "
            f"dist={dados['distancia']:.4f}, ang={dados['angulo']:.2f}°"
        )

    print("\n" + "=" * 60)
    print("GABARITO GERADO COM SUCESSO!")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gera o gabarito de uma figura configurada."
    )
    parser.add_argument(
        "--figura",
        required=True,
        help="Nome da figura em figuras/<nome>/config.json. Exemplo: maca."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gerar_gabarito(caminho_config_figura(args.figura))
