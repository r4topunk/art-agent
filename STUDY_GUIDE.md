# ArtAgent — Study Guide

A complete reference for understanding how ArtAgent autonomously creates pixel art using evolutionary AI.

---

## 1. Visão Geral

ArtAgent é um sistema autônomo que gera pixel art de 16×16 pixels através de evolução iterativa. Nenhum artista humano é necessário — o sistema aprende, gera, avalia e se aprimora sozinho, indefinidamente.

**Filosofia:** Trate a geração de arte como um processo evolutivo. O modelo é o "organismo"; o crítico é a pressão seletiva; cada geração afina o gosto do sistema em direção a peças mais interessantes e coesas.

---

## 2. Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────┐
│                     OvernightRunner                     │
│                                                         │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────┐ │
│  │Bootstrap │───▶│  Trainer  │───▶│    GASLoop       │ │
│  │ (5000    │    │(PixelGPT) │    │                  │ │
│  │patterns) │    └───────────┘    │  generate_pieces │ │
│  └──────────┘         ▲           │  evaluate        │ │
│                        │           │  select          │ │
│  ┌──────────┐          │           │  finetune        │ │
│  │PixelGPT │◀──────────┘           │  save_generation │ │
│  │(decoder) │                      └──────────────────┘ │
│  └──────────┘                            │               │
│       │                                   ▼               │
│  ┌──────────────┐         ┌───────────────────────────┐  │
│  │PixelTokenizer│         │    ArtCritic + VLMCritic  │  │
│  │token↔pixel   │         │  (score & rank pieces)    │  │
│  └──────────────┘         └───────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
                    ┌──────────┐
                    │  TUI     │
                    │(Textual) │
                    └──────────┘
```

**Módulos principais:**

| Arquivo | Responsabilidade |
|---|---|
| `art/config.py` | Hiperparâmetros, paleta de cores, vocab |
| `art/model.py` | PixelGPT — o transformer decoder |
| `art/tokenizer.py` | Conversão pixel ↔ token ↔ imagem |
| `art/data.py` | Bootstrap (5000 padrões) + PixelDataset |
| `art/trainer.py` | Loop de treino com AdamW + AMP |
| `art/critic.py` | Crítico algorítmico (5 dimensões) |
| `art/vlm_critic.py` | Crítico VLM via Ollama/Moondream |
| `art/gas.py` | Loop GAS — o coração do sistema |
| `art/runner.py` | OvernightRunner — orquestrador de alto nível |
| `art/tui/` | Interface visual no terminal |
| `scripts/export_best.py` | Exporta as melhores peças como PNG |

---

## 3. O Modelo: PixelGPT

PixelGPT é um **transformer decoder-only** — a mesma família arquitetural dos modelos de linguagem como GPT, mas aqui cada "palavra" é uma cor de pixel.

### Vocabulário (11 tokens)

| Índice | Token | Significado |
|---|---|---|
| 0–7 | Cores | Paleta de 8 cores com contraste máximo |
| 8 | `BOS` | Begin Of Sequence (início da imagem) |
| 9 | `EOS` | End Of Sequence (fim da imagem) |
| 10 | `PAD` | Padding (alinhamento de batch) |

### Paleta de 8 Cores (contraste máximo)

```
 0: preto       (0, 0, 0)
 1: branco      (255, 255, 255)
 2: vermelho    (255, 0, 0)
 3: azul        (0, 0, 255)
 4: verde       (0, 180, 0)
 5: amarelo     (255, 220, 0)
 6: magenta     (255, 0, 255)
 7: ciano       (0, 220, 220)
```

A paleta reduzida (8 ao invés de 16) simplifica a tarefa do modelo, permite convergência mais rápida, e produz arte com maior contraste visual. Cores foram escolhidas para maximizar distância perceptual.

### Sequência de 258 tokens

Uma imagem 16×16 = 256 pixels → sequência de `[BOS] p₀ p₁ … p₂₅₅ [EOS]` = **258 tokens**.

O modelo lê o prefixo e prevê o próximo token, left-to-right, varrendo a grade de cima para baixo, da esquerda para direita (raster order).

### Hiperparâmetros da Arquitetura

```python
d_model   = 256   # dimensão dos embeddings
n_heads   = 8     # attention heads
n_layers  = 6     # camadas transformer
d_ff      = 1024  # dimensão da feed-forward layer
```

---

## 4. O Ciclo GAS (Genetic Art Selection)

GAS é o loop central do sistema. Cada **geração** executa estas etapas em sequência:

```
Bootstrap ──▶ Treino Inicial
                   │
                   ▼
              ┌─────────┐
         ┌───▶│ GERAÇÃO │  generate_pieces(12 imagens, temperature T)
         │    └────┬────┘
         │         ▼
         │    ┌─────────┐
         │    │AVALIAÇÃO│  critic.score_batch() + (opcional) VLM
         │    └────┬────┘
         │         ▼
         │    ┌─────────┐
         │    │ SELEÇÃO │  3 peças via greedy max-min diversity
         │    └────┬────┘
         │         ▼
         │    ┌──────────┐
         │    │FINETUNE  │  30 steps @ lr=1e-4 + entropy reg
         │    └────┬─────┘
         │         ▼
         │    ┌──────────┐
         └────│ PRÓXIMA  │  generation += 1
              │ GERAÇÃO  │
              └──────────┘
```

### Parâmetros do GAS

```python
images_per_gen    = 12   # peças geradas por geração
select_top        = 3    # selecionadas via diversified selection
finetune_steps    = 30   # passos de finetune por geração (reduzido para evitar overfitting)
finetune_lr       = 1e-4 # learning rate do finetune
bootstrap_mix_ratio    = 0.5  # 50% de bootstrap no finetune
bootstrap_mix_interval = 1    # mistura em toda geração
```

### Seleção Diversificada (anti-colapso)

A seleção não é mais top-k por score. Usa **greedy max-min diversity**:

1. Peça #1: maior composite score
2. Peça #2: maximiza blend de 50% score + 50% distância mínima de Hamming às já selecionadas
3. Peça #3: idem

Isso garante que as 3 peças selecionadas sejam tanto boas quanto diferentes entre si, evitando mode collapse.

---

## 5. Bootstrap: A Semente

Antes da primeira geração, o sistema precisa de conhecimento mínimo sobre estrutura visual. O bootstrap gera **5000 padrões geométricos programáticos** (semente aleatória fixa = 42):

| Padrão | Quantidade aproximada |
|---|---|
| Linhas horizontais | ~80 |
| Linhas verticais | ~80 |
| Linhas diagonais | ~80 |
| Simetria horizontal | ~400 |
| Simetria vertical | ~400 |
| Simetria 4-fold | ~400 |
| Tabuleiros (checkerboard) | ~60 |
| Retângulos preenchidos | ~200 |
| Cruzes | ~150 |
| Diamantes | ~150 |
| Bordas | ~80 |
| Quadrados concêntricos | ~100 |
| Gradientes de cor | ~200 |
| Ruído colorido | ~360 |
| Listras multicoloridas | ~100 |
| Padrões XOR | ~100 |
| Grids de pontos | ~80 |
| Simétricos aleatórios | resto até 5000 |

O modelo é treinado nesses padrões por **80 steps** (treino de bootstrap) antes de começar a evolução. Isso ensina ao PixelGPT conceitos básicos como "pixels adjacentes tendem a ter cores relacionadas".

Em **toda geração**, 50% dos padrões de bootstrap são misturados ao dataset de finetune para evitar catástrofe de esquecimento (*catastrophic forgetting*) e manter diversidade estrutural.

---

## 6. O Crítico Algorítmico

`ArtCritic` avalia cada peça em **5 dimensões** e computa um `composite score`:

```
composite = 0.15 × symmetry
          + 0.25 × complexity
          + 0.20 × structure
          + 0.15 × aesthetics
          + 0.25 × diversity
```

*Se a imagem tiver apenas 1 cor (flat fill), um gate multiplica o composite por 0.3.*

### Dimensão 1: Symmetry (peso 0.15)

Mede simetria em 4 eixos e faz a média:
- Horizontal (`fliplr`)
- Vertical (`flipud`)
- Rotação 180°
- Rotação 90°

Valor = fração de pixels que coincidem após a transformação.

### Dimensão 2: Complexity (peso 0.25)

Três sub-métricas, média simples:

- **Color score** — premia 3–6 cores distintas; penaliza monocromático ou uso de todas as 8 cores
- **Edge score** — fração de transições de cor entre pixels adjacentes (horizontal + vertical)
- **Block entropy** — entropia de Shannon em blocos 4×4, normalizada pelo máximo (`log2(8) = 3`)

### Dimensão 3: Structure (peso 0.20)

Quatro sub-métricas, média simples:

- **Row autocorrelation** — premia coerência moderada nas linhas (sweet spot 0.3–0.7); penaliza ruído ou monocromia
- **Column autocorrelation** — idem para colunas
- **Pattern variety** — número de padrões 2×2 distintos (sweet spot 20–100)
- **Region quality** — flood-fill por cor; premia 5–30 regiões com tamanho mediano de 4–20 pixels

### Dimensão 4: Aesthetics (peso 0.15)

Três sub-métricas, média simples:

- **Balance** — variância das distribuições de cor entre os 4 quadrantes (baixa variância = equilíbrio)
- **Framing** — cor dominante da borda diferente da cor dominante do interior (1.0 vs 0.3)
- **Harmony** — paleta limitada (≤4 cores = 1.0; 5–6 = 0.8; 7–8 = 0.5)

### Dimensão 5: Diversity (peso 0.25)

Calculada **por peça individual** — cada peça recebe sua própria pontuação de novidade:

- Para cada peça, computa a média das distâncias de Hamming para todas as outras peças do batch
- Peças únicas (outliers) recebem scores mais altos; clones recebem scores próximos de zero
- Peso elevado (0.25) para combater mode collapse ativamente

Isso favorece peças que são visualmente distintas do resto do batch, incentivando diversidade real na seleção.

---

## 7. O Crítico VLM

Quando `--vlm` está ativo, cada peça é também avaliada por um modelo de visão local via Ollama (padrão: `moondream`).

### Pipeline

```
Grid (16×16) ──▶ Scale 8× ──▶ PNG base64 ──▶ Ollama API (2 workers)
                 (128×128)                        │
                                                  ▼
                                         "Describe this pixel art
                                          in detail: shapes, patterns,
                                          colors, composition..."
                                                  │
                                                  ▼
                                         _description_to_score()
                                                  │
                                                  ▼
                                     interest / composition / creativity
```

### Extração de Score da Descrição

A resposta textual do VLM é analisada por palavras-chave:

| Tipo | Palavras | Efeito |
|---|---|---|
| Estrutura | shape, pattern, circle, line, cross, geometric… | `composition` ↑ |
| Interesse | interesting, complex, colorful, creative, beautiful… | `interest` ↑ |
| Verbosidade | comprimento da resposta | bônus até +0.3 |
| Negativo | blank, empty, noise, random, static… | penalidade |

```
vlm_composite = 0.4 × interest + 0.3 × composition + 0.3 × creativity
```

### Fusão com o Crítico Algorítmico

```python
composite_final = 0.5 × composite_algoritmico + 0.5 × vlm_composite
```

Quando o VLM não está disponível (Ollama offline), o sistema continua com o crítico algorítmico puro.

---

## 8. Temperature Annealing

A temperatura controla a criatividade do modelo durante a geração:

```
T(g) = temp_start + (g / temp_generations) × (temp_end - temp_start)
     = 1.0       + (g / 50)              × (0.8 - 1.0)
```

| Geração | Temperatura | Comportamento |
|---|---|---|
| 0 | 1.00 | Alta exploração — peças mais diversas/experimentais |
| 25 | 0.90 | Exploração moderada |
| 50+ | 0.80 | Refinamento — peças mais coesas mas ainda com variação |

### Temperatura Reativa (anti-colapso)

Se a diversidade média do batch cair abaixo de 0.15 (distância de Hamming), a temperatura sobe automaticamente para **1.3** na próxima geração, forçando o modelo a explorar novamente. Quando a diversidade se recupera, a temperatura volta ao schedule normal.

**Intuitivamente:** No começo o modelo precisa explorar o espaço de possibilidades. Conforme o modelo aprende o que o crítico valoriza, menor temperatura produz variações mais refinadas. Se o modelo colapsar em um único padrão, o sistema detecta e força re-exploração.

---

## 9. O que a TUI Mostra

A TUI é construída com Textual e tem duas telas: **Dashboard** (padrão) e **Review** (tecla `R`).

### Navegação

| Tecla | Ação |
|---|---|
| `D` | Volta ao Dashboard |
| `R` | Abre tela de Review |
| `Q` | Encerra o programa |

---

### Dashboard Screen

#### Header (topo, largura total)

```
  ArtAgent  │ Gen 7/50  │ Temp 0.86  │ ⏱ 00:23:14  │ Generating...  [│ VLM]
```

- **Gen N/Total** — geração atual e total configurado
- **Temp** — temperatura atual de geração
- **Timer** — tempo decorrido desde o início
- **Phase** — fase atual: `Training (bootstrap)`, `Generating...`, `Selecting & Finetuning...`, `Generation complete`
- **VLM** — badge ciano, visível apenas quando `--vlm` está ativo

#### Training Panel (canto superior esquerdo)

```
TRAINING
──────────────────────
  Phase: finetune
  Step:  45/80
  [█████████████░░░░░░░]
  Loss:  0.8234
  LR:    0.000100

  Loss curve:
  ████▇▆▅▄▃▃▂▂▂▂▂▂▂▁▁▁
```

- Fase do treino (bootstrap ou finetune), progresso em steps, barra de progresso, loss atual, learning rate e sparkline da curva de loss.

#### Gallery Grid (centro superior, todas as 12 peças)

Durante a geração mostra as peças sendo **desenhadas em tempo real** — a cada 16 pixels gerados, a galeria atualiza com o estado parcial de todas as peças. Após a avaliação, exibe todas as 12 peças com seu score composite abaixo de cada uma. As 3 peças selecionadas são destacadas com **borda amarela** e score marcado com `*`.

#### Evolution Panel (canto superior direito)

```
EVOLUTION
──────────────────────────
Mean Score Trend
▂▃▄▄▅▅▆▆▇▇▇█
Max Score Trend
▃▄▅▆▆▇▇▇███

  Mean  0.412
  Max   0.631
  Min   0.201
  Temp  0.860
  Sel   3/12

SCORE BREAKDOWN (avg)
──────────────────────────
  Symm  0.45 ████████░░
  Comp  0.61 ████████████░░░░░░░░
  Stru  0.38 ███████░░░
  Aest  0.52 ██████████░░
  Dive  0.33 ██████░░░░

VLM TREND
▁▂▃▄▄▅
  VLM   0.412
```

- Sparklines de Mean/Max score ao longo das gerações
- Estatísticas da geração atual (mean, max, min, temp, peças selecionadas)
- Score breakdown médio das 5 dimensões com barras coloridas (verde ≥0.6, amarelo ≥0.4, vermelho <0.4)
- Seção VLM (visível apenas quando `--vlm` ativo)

#### Heartbeat Widget (canto inferior esquerdo)

Visualiza o **gradiente de treino** (grad norm) em tempo real como um sinal cardíaco. Também exibe a dificuldade por posição de token — quais posições da sequência o modelo acha mais difíceis de prever.

#### Birth Widget (centro inferior)

Exibe a **melhor peça da geração atual** em destaque, com:
- A imagem ampliada
- Um heatmap de confiança por pixel (quanto o modelo tinha certeza de cada token gerado)
- Durante o treino: preview de uma peça sendo gerada pelo modelo em treinamento
- Se VLM ativo: descrição textual do VLM e scores de interest/composition/creativity

#### Timeline Widget (canto inferior direito)

Linha do tempo horizontal mostrando a **melhor peça de cada geração passada**, da mais antiga para a mais recente. Permite visualizar a evolução estética ao longo do tempo. Se VLM ativo, o score VLM também é exibido abaixo de cada peça.

---

### Review Screen (tecla `R`)

Acessível após a primeira geração ser concluída. Exibe todas as 12 peças da geração atual em uma grade de 8 colunas.

| Tecla | Ação |
|---|---|
| `←` `→` `↑` `↓` | Navega entre as peças |
| `Space` | Marca/desmarca favorito |
| `Esc` | Volta ao Dashboard |

**Detail Panel** (direita): ao selecionar uma peça, exibe:
- Imagem ampliada
- Todos os scores individuais (symmetry, complexity, structure, aesthetics, diversity, composite)
- Se VLM ativo: description, interest, composition, creativity, vlm_composite
- Indicador de favorito

---

### Status Bar (rodapé, largura total)

```
 Gen 7 — Drawing... 62%  │  [D]ash [R]eview [Q]uit
```

Atualiza em tempo real com a fase atual e atalhos de teclado disponíveis.

---

## 10. Peças Finais e Exportação

### Estrutura de Dados por Geração

Cada geração cria o diretório `data/collections/gen_NNN/` com:

```
gen_007/
├── pieces/
│   ├── piece_0000.png   # todas as 12 peças geradas
│   ├── piece_0001.png
│   └── ...
├── scores.json          # scores de cada peça (todas as dimensões)
├── selections.json      # índices das 3 peças selecionadas
└── checkpoint.pt        # estado do modelo após o finetune
```

`scores.json` contém para cada peça:
```json
{
  "symmetry": 0.512,
  "complexity": 0.634,
  "structure": 0.401,
  "aesthetics": 0.489,
  "diversity": 0.321,
  "composite": 0.523,
  "vlm_interest": 0.6,       // se VLM ativo
  "vlm_composition": 0.5,    // se VLM ativo
  "vlm_creativity": 0.55,    // se VLM ativo
  "vlm_composite": 0.553,    // se VLM ativo
  "vlm_description": "..."   // se VLM ativo
}
```

### Retomada Automática

O `OvernightRunner` busca o diretório `gen_NNN` com o maior N e carrega o checkpoint correspondente. Isso permite interromper e retomar a evolução sem perder progresso.

### Script de Exportação

```bash
python scripts/export_best.py [--gen N] [--top 8] [--scale 16]
```

O script:
1. Lê `scores.json` da geração especificada (ou a mais recente)
2. Ordena todas as peças por `composite` score decrescente
3. Pega as top-N (padrão: 8)
4. Escala cada peça 16× (16×16 → 256×256) usando `Image.NEAREST` para manter a estética pixelada
5. Monta uma grade com 4 colunas e padding de 4px sobre fundo `(10, 10, 15)` (quase preto)
6. Salva em `images/best_pieces.png`

Exemplo de saída no console:
```
Exported top 8 pieces from gen 12 to images/best_pieces.png
Scores: #3 (0.631), #17 (0.598), #1 (0.571), #24 (0.554), ...
```

---

## 11. Glossário

| Termo | Definição |
|---|---|
| **GAS** | Genetic Art Selection — o loop evolutivo principal |
| **Bootstrap** | Dataset inicial de 5000 padrões geométricos programáticos |
| **Finetune** | Retreino rápido (30 steps) nas melhores peças da geração, com regularização de entropia |
| **Generation** | Uma iteração completa do loop GAS (gerar → avaliar → selecionar → finetune) |
| **Temperature** | Parâmetro de aleatoriedade do sampling; alto = diverso, baixo = conservador |
| **Composite score** | Score final ponderado que combina as 5 dimensões do crítico |
| **Raster order** | Varredura da imagem linha por linha, da esquerda para a direita |
| **BOS/EOS/PAD** | Tokens especiais: início de sequência, fim de sequência, padding |
| **VLM** | Vision Language Model — modelo de visão local (Moondream via Ollama) |
| **Catastrophic forgetting** | Fenômeno onde o modelo "esquece" conhecimento anterior ao aprender algo novo |
| **Bootstrap mix** | Injeção de padrões do bootstrap em toda geração (50%) para combater catastrophic forgetting |
| **Flood fill** | Algoritmo de preenchimento por contiguidade, usado para contar regiões de cor |
| **Hamming distance** | Fração de pixels diferentes entre duas imagens (0 = idênticas, 1 = completamente diferentes) |
| **Annealing** | Redução gradual da temperatura ao longo das gerações |
| **AMP** | Automatic Mixed Precision — técnica de treino que usa float16 para acelerar sem perder qualidade |
| **EventBus** | Sistema de eventos que conecta o backend (GASLoop, Trainer) à TUI em tempo real |
| **OvernightRunner** | Orquestrador de alto nível: inicializa, retoma, e executa N gerações |
| **PixelGPT** | O transformer decoder-only que gera as sequências de pixels |
| **PixelTokenizer** | Converte imagens RGB ↔ sequências de tokens de cor ↔ grids numpy |
| **Mode collapse** | Fenômeno onde o modelo converge para produzir sempre o mesmo padrão, perdendo diversidade |
| **Greedy max-min** | Algoritmo de seleção que maximiza a distância mínima entre peças selecionadas, garantindo diversidade |
| **Entropy regularization** | Termo na loss que penaliza predições muito confiantes, mantendo diversidade nas gerações |
| **Reactive temperature** | Mecanismo que aumenta a temperatura automaticamente quando a diversidade do batch cai abaixo de um limiar |
| **Per-image diversity** | Score de novidade individual: distância média de Hamming de uma peça para todas as outras do batch |
