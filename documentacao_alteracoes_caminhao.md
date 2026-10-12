# Adaptações da avaliação para as figuras do caminhão e da casa

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

## Alterações gerais, aplicadas às figuras

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

## Figura da casa

### 1. Correção do dataset e treinamento do modelo

O dataset inicial da casa, `figuras_casa.v2i.yolov11`, possuía uma anotação de
treinamento incorreta na imagem `IMG_5065_HEIC`: uma região de `tijolos` foi
rotulada como `telhado`. Como consequência, a V2 possuía 39 ocorrências de
`telhado` e 37 de `tijolos`, enquanto as outras cinco classes tinham 38
ocorrências cada.

A versão `figuras_casa.v3i.yolov11` corrigiu somente essa classe, mantendo as
mesmas imagens, a mesma divisão de dados e a mesma geometria dos polígonos.
Com isso, as sete classes (`chamine`, `degrau`, `entrada`, `jardim`, `porta`,
`telhado` e `tijolos`) passaram a ter 38 ocorrências cada. O conjunto contém
31 imagens de treinamento, 7 de validação e 6 de teste.

O modelo de segmentação foi treinado com a V3 e salvo como `best-casa.pt`. A
melhor época foi a 49, com precisão de máscara de aproximadamente 99,0%,
recall de 99,9%, mAP@50 de 99,5% e mAP@50--95 de 97,4% na validação. Apesar
das métricas altas, esse resultado deve ser interpretado com cautela devido ao
conjunto de validação pequeno e visualmente semelhante às imagens de treino.

### 2. Configuração da regra de pontuação da casa

A casa contém sete peças e dez junções pontuáveis, enquanto o caminhão possui
quatro peças e cinco junções e a maçã possui três peças e três junções. Foram
cadastradas as seguintes relações no `config.json` da casa:

| Junção | Peças |
| --- | --- |
| `chamine_telhado` | chaminé e telhado |
| `chamine_porta` | chaminé e porta |
| `porta_jardim` | porta e jardim |
| `telhado_porta` | telhado e porta |
| `telhado_tijolos` | telhado e tijolos |
| `porta_tijolos` | porta e tijolos |
| `tijolos_entrada` | tijolos e entrada |
| `tijolos_degrau` | tijolos e degrau |
| `porta_entrada` | porta e entrada |
| `entrada_degrau` | entrada e degrau |

Assim, a nota máxima da casa é 10. O gabarito foi construído a partir da
imagem `casa-correta.png`, na qual as sete peças foram detectadas uma única
vez com confiança entre 87,3% e 99,6%.

### 3. Adaptações exclusivas da casa para contatos deslocados

Na avaliação humana do sujeito 5, chaminé e telhado estavam deslocados
horizontalmente. O avaliador original detectava contato físico e aprovava
indevidamente as junções `chamine_porta` e `telhado_tijolos`. Isso é o mesmo
tipo de limitação geométrica encontrado no caminhão: a existência de contato
não garante que o contato ocorreu no trecho correto da peça.

Para a casa, foi reutilizado o mecanismo genérico de posição local do contato,
sem alterar novamente os scripts e sem alterar maçã ou caminhão. Foram
habilitadas somente estas tolerâncias:

| Junção | Tolerância de posição local |
| --- | ---: |
| `chamine_porta` | 0,12 |
| `telhado_tijolos` | 0,25 |

Também foi criado um contato mínimo específico de 0,22 para
`telhado_porta`. Esse valor aceita os sujeitos humanos corretos 2 e 10, cujos
contatos medidos foram 0,24 e 0,23, sem aceitar o sujeito 5, cujo contato foi
0,06.

Como a checagem de posição precisa da referência espacial da junção, o
gabarito da casa foi regenerado no esquema 3, contendo os centróides dos
contatos. A maçã continua no comportamento anterior, e as regras específicas
do caminhão continuam definidas somente no arquivo de configuração daquele
item.

### 4. Comparação com a avaliação humana e limitações observadas

Após as adaptações de contato, as notas automáticas foram 0, 10, 0, 0, 7, 0,
0, 0, 10 e 10 para os sujeitos 1 a 10, respectivamente. As notas humanas foram
0, 10, 1, 1, 7, 1, 1, 2, 10 e 10. Houve coincidência numérica nos sujeitos 1,
2, 5, 9 e 10; no sujeito 1, porém, a coincidência ocorreu apesar de erro de
detecção, pois a avaliação automática não encontrou todas as peças e atribuiu
zero.

As divergências restantes não foram causadas pela regra de junção:

- Nos sujeitos 4, 6, 7 e 8, uma ou mais peças não foram detectadas. Sem as
  sete máscaras, a avaliação é interrompida para evitar pontuação baseada em
  uma montagem incompleta.
- No sujeito 3, as sete classes foram listadas, mas algumas máscaras ficaram
  incorretas em uma foto com peças muito afastadas, rotacionadas e parcialmente
  sobrepostas. A única junção humana correta, `porta_jardim`, foi medida sem
  contato e com distância incompatível com o gabarito.

Esses casos revelam uma limitação do modelo, e não um limiar a ser reduzido no
avaliador. O treinamento da casa foi composto principalmente por montagens
corretas; com poucas imagens, o modelo também aprende as posições típicas das
peças. Para aumentar a robustez, o dataset deve receber imagens anotadas de
peças isoladas, deslocadas, rotacionadas e parcialmente sobrepostas, seguidas
de um novo treinamento.

### 5. Diferenças da casa em relação ao caminhão

O caminhão exigiu tolerâncias de contato específicas para todas as cinco
junções e uma estimativa robusta de rotação global pela mediana das orientações
das peças. A casa precisou de contato específico apenas para
`telhado_porta` e de posição de contato em dois pares. Até a avaliação atual,
não foi necessário ativar na casa o método de mediana para rotação global;
seu principal problema observado foi a ausência ou a segmentação incorreta de
peças, não uma rotação global calculada de forma incorreta.

As configurações são independentes por figura. Portanto, as adaptações da casa
não modificam os critérios, gabaritos, pesos ou notas da maçã e do caminhão.

## Síntese

As regras adicionais não tornam o algoritmo dependente do caminhão: elas são
recursos opcionais e configuráveis por figura. O caminhão as habilita porque o
contato físico isolado não garante um encaixe correto em suas bordas longas. A
maçã não as habilita porque a sua geometria já é adequadamente avaliada pelos
critérios de distância, ângulo, rotação e contato mínimo global.
