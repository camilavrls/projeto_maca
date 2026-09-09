# Adaptações da avaliação para a figura do caminhão

## Contexto

O processo de avaliação é genérico: o modelo segmenta cada peça e o programa
compara a montagem fotografada com um gabarito de uma montagem correta. As
regras de distância entre centróides, ângulo relativo, rotação das peças e
contato entre máscaras são utilizadas para decidir se cada junção está correta.

Embora a estrutura do programa seja compartilhada, o caminhão exigiu regras
adicionais que não foram habilitadas para a maçã. A razão é geométrica: a
figura do caminhão possui peças com bordas longas em comum. Assim, uma peça
deslocada ainda pode tocar outra máscara e produzir uma medida alta de contato,
mesmo que o encaixe tenha ocorrido no trecho errado. Na maçã, as junções são
mais curtas e específicas; ao deslocar uma peça, normalmente surge uma lacuna
ou mudam a distância, o ângulo e a rotação, de modo que os critérios originais
já distinguem a montagem incorreta.

## Alterações exclusivas do caminhão

### 1. Contato mínimo específico para cada junção

No arquivo `figuras/caminhao/config.json`, foi utilizado o campo
`contato_min_por_par`. Ele permite definir o mínimo de contato de máscara para
cada par de peças, em vez de aplicar um único valor a todas as junções.

| Junção | Contato mínimo |
| --- | ---: |
| `baixo_cima` | 0,25 |
| `baixo_direita` | 0,25 |
| `baixo_esquerda` | 0,25 |
| `cima_direita` | 0,22 |
| `cima_esquerda` | 0,10 |

Essa diferenciação foi necessária porque as áreas e os formatos dos contatos
do caminhão não são homogêneos. A junção `cima_esquerda` possui uma área menor
que as demais, mas os testes mostraram que 0,10 ainda aceita as montagens
corretas e evita considerar como encaixe um contato residual muito pequeno.
A maçã mantém um único parâmetro global, `contato_min: 0,24`, pois suas junções
possuem comportamento suficientemente semelhante para esse critério.

### 2. Verificação da posição do contato em pares selecionados

O caminhão habilita `contato_posicao_local_por_par` para as junções
`baixo_cima`, `baixo_direita`, `cima_direita` e `cima_esquerda`. Além de verificar se duas
máscaras encostam, o avaliador verifica se o contato ocorre na região esperada
do par de peças, usando como referência a montagem correta.

As tolerâncias configuradas são:

| Junção | Tolerância de posição local |
| --- | ---: |
| `baixo_cima` | 0,10 |
| `baixo_direita` | 0,31 |
| `cima_direita` | 0,08 |
| `cima_esquerda` | 0,10 |

A posição é calculada no sistema de coordenadas formado pelos centróides das
duas peças da própria junção. Por isso, a comparação é resistente a
deslocamentos, escala e rotação global da fotografia. Essa regra corrige o
caso em que a peça direita do caminhão está deslocada: ela ainda pode tocar a
peça de baixo, mas o toque acontece em uma região diferente da registrada no
gabarito. Ela também evita aprovar `cima_esquerda` quando há contato, mas ele
ocorre no trecho errado da borda.

Essa verificação não foi ativada para a maçã. Seu `config.json` não contém
`contato_posicao_local_por_par`; portanto, a avaliação da maçã continua usando
somente os quatro critérios originais. O código só executa o quinto critério
quando a chave está presente na configuração da figura.

### 3. Gabarito do caminhão com posição dos contatos

Como a verificação de posição exige uma referência, o arquivo
`figuras/caminhao/gabarito.npz` foi regenerado no formato de esquema 3. Além
dos centróides, ângulos e relações entre pares, ele armazena o centróide da
região de contato de cada junção, no campo `pares_contato_centroides`.

O gabarito da maçã não precisa desse campo e permanece compatível com o formato
anterior. O carregamento é retrocompatível: se o campo não existir, a regra de
posição não é aplicada.

### 4. Estimativa robusta da rotação global da fotografia

O caminhão utiliza `"metodo": "mediana_angulos_pecas"` em
`rotacao_global`. Antes dessa alteração, a rotação global era estimada pela
posição dos centróides das peças. Em montagens parcialmente incorretas, as
peças deslocadas alteravam essa geometria e produziam uma rotação global
errada. Como consequência, uma junção visualmente correta, como
`baixo_esquerda`, podia ser reprovada apenas pelos critérios de ângulo e
rotação.

Com o novo método, o programa calcula a diferença de orientação de cada peça
em relação ao gabarito e usa a mediana dessas diferenças. A mediana é robusta
a valores discrepantes: uma peça mal posicionada não determina a rotação da
foto. Essa adaptação foi necessária no caminhão porque suas avaliações incluem
montagens com várias peças deslocadas, que comprometem a referência geométrica
global.

A maçã mantém o método anterior, baseado nos centróides, pois essa situação
não foi observada em suas imagens avaliadas. Como o método é selecionado no
`config.json` do caminhão, não modifica o cálculo usado para a maçã.

## Alterações gerais, aplicadas às duas figuras

As alterações a seguir não são específicas do caminhão e, por isso, foram
adotadas para maçã e caminhão:

1. **Confiança separada por etapa.** Cada configuração possui
   `confianca_gabarito: 0.70` e `confianca_avaliacao: 0.10`. O gabarito exige
   detecções muito confiáveis, pois define a referência. Na avaliação, o
   limiar menor evita que uma peça existente seja tratada como ausente apenas
   por variações de iluminação, enquadramento ou segmentação.
2. **Imagem no formato de cor correto para o YOLO.** A inferência passou a
   receber a imagem no padrão BGR carregado pelo OpenCV. A conversão para RGB é
   mantida apenas quando necessária para exibição, evitando a alteração de
   cores na entrada do modelo.

## Síntese

As regras adicionais não tornam o algoritmo dependente do caminhão: elas são
recursos opcionais e configuráveis por figura. O caminhão as habilita porque o
contato físico isolado não garante um encaixe correto em suas bordas longas. A
maçã não as habilita porque a sua geometria já é adequadamente avaliada pelos
critérios de distância, ângulo, rotação e contato mínimo global.
