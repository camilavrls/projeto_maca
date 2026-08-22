# Contexto Para Agentes

Este repositório contém um protótipo em Python para segmentar peças de uma montagem com YOLO e avaliar automaticamente se as junções da montagem estão corretas. O domínio descrito no `README.md` é a correção de testes aplicados em crianças de 3 a 10 anos.

## Visão Geral

- Modelo principal: `best-seg.pt`, carregado via `ultralytics.YOLO`.
- Script de segmentação visual: `segmentar.py`.
- Scripts de avaliação por junções: `scripts_juncao/gerar_gabarito_juncoes.py` e `scripts_juncao/avaliar_montagem.py`.
- Configuração principal da figura da maçã: `figuras/maca/config.json`.
- Gabarito atual da maçã: `figuras/maca/gabarito.npz`.
- Imagem correta usada para gerar o gabarito da maçã: `figuras/maca/correta.jpg`.
- Imagens avaliadas em lote para a maçã: `figuras/maca/imagens_avaliacao/`.

## Estrutura

```text
.
├── README.md
├── AGENTS.md
├── best-seg.pt
├── figuras/
│   └── maca/
│       ├── config.json
│       ├── gabarito.npz
│       ├── correta.jpg
│       ├── incorreta2.jpg
│       └── imagens_avaliacao/
├── segmentar.py
└── scripts_juncao/
    ├── avaliar_montagem.py
    └── gerar_gabarito_juncoes.py
```

## Dependências

Não há arquivo `requirements.txt` ou `pyproject.toml` no estado atual do repositório. Pelos imports, os scripts esperam pelo menos:

- `ultralytics`
- `opencv-python` ou pacote equivalente que forneça `cv2`
- `numpy`
- `openpyxl`

Antes de executar, confirme que essas dependências estão instaladas no ambiente Python ativo.

## Como Executar

### Segmentação visual simples

Na raiz do projeto:

```bash
python segmentar.py
```

Por padrão, o bloco `if __name__ == "__main__"` chama:

```python
segmentar("figuras/maca/correta.jpg")
```

Saída esperada:

- Pasta criada automaticamente: `resultado_segmentacao/`
- Arquivo de imagem com sufixo `_segmentada.jpg`

### Gerar gabarito de junções

Na raiz do projeto:

```bash
python3 scripts_juncao/gerar_gabarito_juncoes.py --figura maca
```

Esse script:

- Carrega `figuras/maca/config.json`.
- Carrega o modelo configurado em `modelo`.
- Segmenta a imagem configurada em `imagem_gabarito`.
- Calcula centroides normalizados, ângulos de orientação e relações entre os pares configurados.
- Salva o gabarito no caminho configurado em `gabarito`.

### Avaliar imagens em lote

Pode ser executado a partir da raiz do projeto:

```bash
python3 scripts_juncao/avaliar_montagem.py --figura maca
```

Para a maçã, a chamada interna equivalente é:

```python
avaliar_pasta(caminho_config_figura("maca"))
```

O argumento `--figura maca` faz o script carregar `figuras/maca/config.json`. Os caminhos declarados nesse JSON são resolvidos em relação à raiz do projeto.

Saída esperada:

- Logs no terminal com nota final por imagem.
- Planilha `notas_montagens.xlsx` em `figuras/maca/`.

## Modelo de Pontuação

`scripts_juncao/avaliar_montagem.py` lê as classes esperadas de `figuras/maca/config.json`. Para a figura atual da maçã, são:

- `Esquerda`
- `Direita_cima`
- `Direita_baixo`

A nota vai de `0/N` a `N/N`, onde `N` é o número de pares configurados. Para a maçã atual, `N = 3`, correspondendo aos pares:

- `esquerda_cima`
- `esquerda_baixo`
- `cima_baixo`

Cada par passa apenas se todos os critérios abaixo passarem:

- Distância relativa entre centroides dentro de `TOL_DIST_FRAC`.
- Ângulo relativo compatível com o gabarito, compensando rotação global.
- Rotação individual das peças dentro de `TOL_ROT_DEG`.
- Contato real entre máscaras acima de `TOL_CONTATO_MIN`.

Os limites atuais em `avaliar_montagem.py` são:

```json
"confianca_minima": 0.10,
"tolerancias": {
  "distancia_frac": 0.20,
  "angulo_graus": 50.0,
  "rotacao_graus": 30.0,
  "contato_min": 0.25
}
```

## Detalhes Importantes

- `segmentar.py` carrega `best-seg.pt` com caminho relativo à raiz do repositório.
- `avaliar_montagem.py` exige `--figura` e carrega `figuras/<figura>/config.json`.
- Os caminhos de `avaliar_montagem.py` são resolvidos em relação à raiz do projeto.
- `gerar_gabarito_juncoes.py` também recebe `--figura` e carrega `figuras/<figura>/config.json`.
- `gerar_gabarito_juncoes.py` não deduplica classes; `avaliar_montagem.py` mantém apenas a detecção de maior confiança por classe.
- `avaliar_montagem.py` usa `DEBUG = False` por padrão. Ao ativar `DEBUG = True`, a planilha ganha colunas técnicas extras e o terminal mostra detalhes dos critérios.
- O formato novo do `gabarito.npz` é genérico: guarda nomes dos pares, classes de cada par, distâncias e ângulos em arrays, sem campos fixos da maçã.
- Há textos com emojis nos logs dos scripts existentes. Manter ou remover deve ser uma decisão explícita, pois muda a experiência atual do terminal.

## Cuidados Para Alterações Futuras

- Evite mudar caminhos relativos sem ajustar os comandos e os scripts chamadores.
- Se mover `best-seg.pt`, atualize `CAMINHO_PESOS` em `segmentar.py` e `modelo` em `figuras/maca/config.json`.
- Se alterar nomes de classes no modelo, atualize `classes_esperadas`, `pares` e `rotacao_global` em `figuras/maca/config.json`.
- Se trocar o gabarito, gere novamente `gabarito.npz` a partir da imagem correta correspondente.
- Se adicionar novas imagens de avaliação, use extensões reconhecidas por `EXTENSOES_IMAGEM`.
- Se mexer nos critérios de avaliação, registre a justificativa, pois pequenas mudanças em tolerâncias podem alterar notas finais. As tolerâncias da maçã ficam em `figuras/maca/config.json`.
- Como ainda não há suíte de testes, valide mudanças executando ao menos a avaliação em lote e conferindo a planilha gerada.

## Ao Adicionar Novas Figuras

- Crie uma pasta `figuras/<nome>/` com `config.json`, `correta.jpg`, `gabarito.npz` e `imagens_avaliacao/`.
- Gere o gabarito com `python3 scripts_juncao/gerar_gabarito_juncoes.py --figura <nome>` antes de avaliar.
- Avalie com `python3 scripts_juncao/avaliar_montagem.py --figura <nome>`.
- O código suporta figuras com mais de 3 peças; a nota máxima vem de `len(pares)` no `config.json`.
- Cada item em `pares` deve ligar exatamente duas classes. Junções envolvendo mais peças devem ser modeladas como múltiplos pares.
- A imagem de gabarito deve detectar exatamente as classes listadas em `classes_esperadas`; o gerador falha se houver classe ausente, extra ou duplicada.
- O avaliador deduplica detecções por classe mantendo a maior confiança; o gerador ainda não deduplica para evitar salvar um gabarito ambíguo.
- `rotacao_global` é obrigatório no formato atual e precisa apontar para classes existentes na figura. Escolha uma origem e destino que representem bem a orientação geral da figura.
- A estratégia atual de rotação global calcula um vetor da classe `origem` até o centroide médio de `destino_centroide`. Se uma nova figura não tiver um eixo visual claro, pode ser necessário evoluir essa estratégia.
- O modelo configurado em `modelo` precisa ter sido treinado para reconhecer as classes da nova figura.

## Melhorias Pendentes Recomendadas

- Criar `requirements.txt` ou `pyproject.toml`.
- Adicionar argumentos de linha de comando para caminhos de imagem, pasta, modelo, gabarito e saída.
- Separar utilitários compartilhados de configuração/geometria para evitar duplicação entre gerador e avaliador.
- Separar lógica de avaliação de efeitos colaterais de terminal/Excel para facilitar testes.
- Adicionar testes unitários para cálculo de ângulos, distância, rotação global, deduplicação e status final.
- Documentar versão do modelo YOLO e classes treinadas.
