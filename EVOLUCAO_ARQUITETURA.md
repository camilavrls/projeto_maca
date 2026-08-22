# Evolução Para Arquitetura Agnóstica

Este documento resume as mudanças feitas para preparar o projeto para avaliar diferentes figuras de quebra-cabeça, não apenas a maçã.

## Antes

- Os dados da maçã ficavam misturados com os scripts em `scripts_juncao/`.
- Classes, pares avaliados e nota máxima estavam fixos no código.
- O gabarito `.npz` tinha campos específicos da maçã, como pares ligados a `esquerda_cima`, `esquerda_baixo` e `cima_baixo`.
- A execução não indicava explicitamente qual figura estava sendo avaliada.

## Depois

- Os dados da figura ficam em uma pasta própria:

```text
figuras/
└── maca/
    ├── config.json
    ├── gabarito.npz
    ├── correta.jpg
    └── imagens_avaliacao/
```

- Os scripts principais passaram a receber a figura por argumento:

```bash
python3 scripts_juncao/gerar_gabarito_juncoes.py --figura maca
python3 scripts_juncao/avaliar_montagem.py --figura maca
```

- A configuração da figura define:
  - modelo YOLO
  - imagem correta para geração do gabarito
  - caminho do gabarito
  - pasta de imagens avaliadas
  - classes esperadas
  - pares/junções avaliadas
  - tolerâncias
  - referência para compensação de rotação global

- O gabarito passou a ter formato genérico (`schema_version = 2`), salvando pares como listas/arrays:
  - `pares_nomes`
  - `pares_classes`
  - `pares_distancias`
  - `pares_angulos`

## Benefício

Para adicionar uma nova figura, como caminhão ou casa, a ideia passa a ser criar uma nova pasta em `figuras/<nome>/` com seu próprio `config.json`, imagem correta, imagens de avaliação e gabarito. Os scripts continuam os mesmos.

## Pontos De Atenção

- O modelo precisa reconhecer as classes da nova figura.
- Cada par configurado deve ter exatamente duas classes.
- A imagem usada para gerar o gabarito deve detectar todas as classes esperadas, sem classes ausentes, extras ou duplicadas.
- A rotação global ainda usa uma estratégia única: vetor entre uma peça de origem e o centroide de uma ou mais peças destino.
- Figuras com mais peças são suportadas, desde que `classes_esperadas`, `pares` e `rotacao_global` estejam bem definidos no `config.json`.
