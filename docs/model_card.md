# Model Card — Classificador de Urgência de Laudos (Fase 3)

## Descrição

Classificador de texto (TF-IDF + Regressão Logística) que recebe o texto de
um laudo/resumo médico e retorna um nível de urgência fictício: `normal`,
`atencao` ou `urgente`. Construído para o Tech Challenge Fase 3
(Pós-Tech/MLET), cujo tema central é o ciclo de vida do modelo em produção
(deploy, CI/CD, orquestração, monitoramento, latência) — não a
sofisticação do modelo em si.

## Dados de treino

- **Fonte**: [Medical Abstracts TC Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)
  (Schopf, Braun & Matthes, 2022), CC BY-SA 3.0.
- **Tamanho**: 11.550 amostras de treino, 2.888 de teste (14.438 no total).
- **Texto original**: resumos de literatura médica (não laudos clínicos
  reais de pronto-socorro).
- **Rótulo original**: 5 categorias de condição clínica (neoplasias,
  doenças digestivas, doenças do sistema nervoso, doenças cardiovasculares,
  condições patológicas gerais).

## ⚠️ Limitação central: o rótulo de urgência é uma simplificação didática

O corpus **não tem** rótulo de urgência. Para atender ao cenário do
desafio, mapeamos as 5 categorias originais em 3 níveis de urgência com uma
regra documentada em `src/triage/data/labels.py`:

| Nível | Categorias originais | Critério |
|---|---|---|
| `urgente` | Neoplasias, doenças cardiovasculares | Associadas na literatura a quadros agudos/sistêmicos que tipicamente exigem investigação rápida |
| `atencao` | Doenças digestivas, condições patológicas gerais | Evolução mais variável, nem sempre emergencial |
| `normal` | Doenças do sistema nervoso | Tratada como classe de referência de menor prioridade **apenas neste recorte didático** |

**Isso não é uma regra clínica validada.** Doenças do sistema nervoso
incluem quadros que são, na realidade, emergências (ex.: AVC) — o
agrupamento serve exclusivamente para produzir 3 classes plausíveis o
suficiente para exercitar o pipeline de MLOps pedido pela fase (API, CI/CD,
Airflow, monitoramento, otimização de latência). **O modelo não deve ser
usado para triagem clínica real.**

## Distribuição das classes

Idêntica em treino e teste (mapeamento determinístico de um dataset já
dividido pelos autores originais):

| Classe | Proporção |
|---|---|
| `atencao` | 43.6% |
| `urgente` | 43.0% |
| `normal` | 13.3% |

Desbalanceamento moderado — mitigado com `class_weight="balanced"` na
Regressão Logística (ver `src/triage/models/train.py`).

## Arquitetura do modelo

- **Vetorização**: `TfidfVectorizer` (unigramas + bigramas, até 20.000
  features, `sublinear_tf=True`).
- **Classificador**: `LogisticRegression` (`class_weight="balanced"`,
  `max_iter=1000`).
- **Seed fixada**: 42, em todos os componentes com aleatoriedade.

## Métricas (conjunto de teste, 2.888 amostras)

| Modelo | Macro-F1 |
|---|---|
| Classe majoritária (baseline ingênuo) | 0.2025 |
| Naive Bayes (TF-IDF) | 0.5374 |
| **TF-IDF + Regressão Logística (modelo principal)** | **0.6347** |

```
              precision    recall  f1-score   support
     atencao       0.68      0.61      0.64      1260
      normal       0.50      0.68      0.57       385
     urgente       0.69      0.69      0.69      1243
    accuracy                           0.65      2888
   macro avg       0.62      0.66      0.63      2888
weighted avg       0.66      0.65      0.65      2888
```

**Leitura honesta**: a classe `normal` tem a menor precisão (0.50) —
consequência direta de ser a classe minoritária (13.3%) mapeada a partir de
uma única categoria original (doenças do sistema nervoso), com maior
sobreposição de vocabulário com as outras classes do que se pode dizer o
oposto de fora do agrupamento. Isso está diretamente ligado à limitação de
rotulagem descrita acima, não a um defeito de implementação.

## Otimização de latência

Classificador exportado para ONNX Runtime (`src/triage/models/optimize.py`)
— ganho medido de ~1.10-1.14x (p95/p50) sobre o pipeline scikit-learn
original. Ver [`../reports/latency_comparison.md`](../reports/latency_comparison.md)
para a discussão completa, incluindo por que o ganho é modesto (apenas o
estágio de classificação foi convertido, não a vetorização TF-IDF).

## Riscos e usos não recomendados

- **Não é um dispositivo médico** e não passou por nenhuma validação
  clínica, regulatória ou de segurança do paciente.
- O rótulo de urgência é uma construção didática (ver acima) — usar este
  modelo para decisões reais de triagem seria um mau uso sério.
- O texto de entrada em produção real (laudos de pronto-socorro) tem estilo
  e vocabulário diferentes de resumos de artigos científicos (o dado de
  treino) — esperado *drift* significativo fora deste contexto acadêmico.
- Dataset em inglês; não testado com laudos em português.

## Reprodutibilidade

```bash
uv sync
uv run python -m triage.models.train      # treina e salva o modelo
uv run python -m triage.models.optimize   # exporta ONNX + benchmark de latência
uv run pytest tests/ -v                   # 14 testes, cobrindo dados, modelo, API
```
