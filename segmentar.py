from ultralytics import YOLO
import cv2
from pathlib import Path

CAMINHO_PESOS = "best-seg.pt"
PASTA_SAIDA = "resultado_segmentacao"
CONF_MINIMA = 0.25

EXTENSOES_VALIDAS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

model = YOLO(CAMINHO_PESOS)

def segmentar(caminho_imagem):
    caminho_imagem = Path(caminho_imagem)

    imagem = cv2.imread(str(caminho_imagem))

    if imagem is None:
        print(f"Imagem não encontrada ou inválida: {caminho_imagem}")
        return

    resultados = model(str(caminho_imagem), conf=CONF_MINIMA)
    resultado = resultados[0]

    imagem_segmentada = resultado.plot()

    Path(PASTA_SAIDA).mkdir(parents=True, exist_ok=True)

    nome_saida = caminho_imagem.stem + "_segmentada.jpg"
    caminho_saida = Path(PASTA_SAIDA) / nome_saida

    cv2.imwrite(str(caminho_saida), imagem_segmentada)

    print(f"Imagem segmentada salva em: {caminho_saida}")


def segmentar_pasta(caminho_pasta):
    caminho_pasta = Path(caminho_pasta)

    if not caminho_pasta.exists():
        print(f"❌ Pasta não encontrada: {caminho_pasta}")
        return

    if not caminho_pasta.is_dir():
        print(f"❌ O caminho informado não é uma pasta: {caminho_pasta}")
        return

    imagens = [
        arquivo for arquivo in caminho_pasta.iterdir()
        if arquivo.suffix.lower() in EXTENSOES_VALIDAS
    ]

    if not imagens:
        print(f"⚠️ Nenhuma imagem encontrada na pasta: {caminho_pasta}")
        return

    print(f"📁 {len(imagens)} imagem(ns) encontrada(s) em: {caminho_pasta}")

    for imagem in imagens:
        segmentar(imagem)


if __name__ == "__main__":

    # segmentar("imagens_avaliacao/sujeito7.jpg")
    segmentar("scripts_juncao/correta.jpg")

    # segmentar_pasta("scripts_juncao/imagens_avaliacao")