# medical-triage-mlops

Tech Challenge — Fase 3 (Pós-Tech / MLET): **Deploy de Modelo em Produção com
Pipeline CI/CD, Monitoramento e Otimização de Latência**.

Cenário do enunciado: um hospital fictício precisa de um sistema de triagem
automática de laudos médicos, classificando-os em três níveis de urgência
(`normal` / `atencao` / `urgente`). O projeto cobre o ciclo completo pedido
pela fase: API de inferência, pipeline de CI/CD, orquestração de retreino,
observabilidade e otimização de latência.

> ⚠️ **Aviso importante**: este é um projeto acadêmico com dados públicos de
> literatura médica (não laudos reais de pronto-socorro) e uma regra de
> urgência **fictícia e não validada clinicamente** — ver "Dataset e
> rótulos" abaixo. Não deve ser usado para triagem clínica real.

---

## Sumário

- [Decisão arquitetural (deploy em nuvem)](#decisão-arquitetural-deploy-em-nuvem)
- [Arquitetura do projeto](#arquitetura-do-projeto)
- [Dataset e rótulos](#dataset-e-rótulos)
- [Execução local](#execução-local)
- [CI/CD](#cicd)
- [Orquestração (Airflow)](#orquestração-airflow)
- [Monitoramento](#monitoramento)
- [Otimização de latência](#otimização-de-latência)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Resultados](#resultados)

---

## Decisão arquitetural (deploy em nuvem)

O enunciado pede uma análise textual de **batch vs. real-time** e da nuvem
escolhida (AWS, Azure ou GCP).

**Natureza da carga**: cada laudo chega isoladamente, no momento em que é
produzido, e precisa de uma resposta em segundos para apoiar a triagem — não
há um lote acumulado para processar periodicamente. Isso descarta batch como
padrão principal (poderia existir como um job auxiliar de re-scoring
noturno, mas não como o caminho primário de servir o modelo).

**Real-time via container HTTP stateless** é a escolha: a API FastAPI já
empacotada em Docker (ver `Dockerfile`) é o próprio artefato de deploy —
não precisa de re-empacotamento para produção.

**Provedor**: **GCP, com Cloud Run**, por continuidade com as Fases 1 e 2
(mesmo projeto `fiap-fase1`, já provisionado, com billing e APIs ativos) e
porque o modelo do trade-off aplicado nas fases anteriores continua válido
aqui:

| Critério | Cloud Run (GCP) | App Service (Azure) | ECS Fargate (AWS) |
|---|---|---|---|
| Escala a zero (custo ocioso) | ✅ nativo | parcial (planos específicos) | não (tasks sempre ativas) |
| Deploy direto de imagem Docker | ✅ | ✅ | ✅ (mais configuração: task def, ALB) |
| Fricção operacional para 1 serviço pequeno | baixa | média | média-alta |
| Reaproveita histórico do time | ✅ (Fases 1 e 2 já rodam lá) | — | — |

Cloud Run também escala a zero entre picos de uso — relevante para um
serviço de triagem que não tem tráfego constante 24/7 num ambiente de
estudo — e o mesmo container local (`docker build` + `docker compose up`)
é o que subiria em produção, sem adaptação. A stack de monitoramento local
(Prometheus + Grafana, exigida pelo enunciado) seria substituída em produção
por Cloud Monitoring/Managed Prometheus do GCP, mas os mesmos endpoints
`/metrics` continuam funcionando como fonte.

*(O deploy em nuvem em si é bônus/fora do escopo obrigatório desta fase —
aqui documentamos a decisão, como pedido; a Fase 2 já demonstrou esse passo
funcionando de ponta a ponta com o mesmo projeto GCP.)*

---

## Arquitetura do projeto

```
Cliente → FastAPI (/triage) → TriagePredictor (sklearn ou ONNX Runtime)
                ↓
        prometheus-client (/metrics) → Prometheus → Grafana (dashboards)

Airflow DAG (medical_triage_retrain):
  load_data → train_and_save_model → report_metrics
  (mesmas funções de src/triage/, testadas independente do Airflow)

GitHub Actions (ci.yml):
  lint (ruff) → test (pytest, treina o modelo) → build (docker build)
```

O modelo é intencionalmente leve (TF-IDF + Regressão Logística): o tema da
fase é o **ciclo de vida em produção**, não a sofisticação do classificador
— mesma lógica de priorização usada nas Fases 1 e 2 (lá, infraestrutura
pesava mais que o modelo nos critérios de avaliação).

---

## Dataset e rótulos

**Fonte**: [Medical Abstracts TC Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)
(Schopf, Braun & Matthes, 2022), CC BY-SA 3.0 — 14.438 resumos de literatura
médica (11.550 treino / 2.888 teste), muito acima do mínimo de 2.000
amostras pedido pelo enunciado. Commitado em `data/raw/` (~17 MB, pequeno o
suficiente para não precisar de DVC/Git LFS).

O corpus rotula cada resumo em **5 categorias de condição clínica**
(neoplasias, doenças digestivas, doenças do sistema nervoso, doenças
cardiovasculares, condições patológicas gerais) — **não** é um dataset de
laudos de pronto-socorro com rótulo de urgência real.

Para o cenário fictício do desafio, `src/triage/data/labels.py` agrupa as 5
categorias em 3 níveis de urgência (`normal` / `atencao` / `urgente`) com um
critério **documentado e explicitamente não-clínico** (categorias
tipicamente associadas a quadros agudos → "urgente"; ver o docstring do
módulo para a regra completa e a ressalva). Isso é uma simplificação
didática para exercitar o pipeline de MLOps, não uma validação médica —
discutido também em `docs/model_card.md`.

---

## Execução local

Requer [uv](https://docs.astral.sh/uv/) e Python 3.11+.

```bash
uv sync                                    # cria .venv e instala do uv.lock
uv run python -m triage.models.train       # treina e salva models/triage_pipeline.joblib
uv run pytest tests/ -v --cov=src          # roda a suíte de testes (14 testes)
uv run ruff check src tests                # lint

uv run uvicorn triage.api.main:app --reload   # sobe a API em http://localhost:8000/docs
```

Exemplo de chamada:

```bash
curl -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient presents with acute chest pain and diaphoresis."}'
```

### Stack completa (API + Prometheus + Grafana)

```bash
docker compose up --build
```

- API: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (login `admin`/`admin`, ou acesso anônimo
  já habilitado) — dashboard "Triagem API" provisionado automaticamente.

Validado de ponta a ponta em ambiente local (Windows 11 + Docker Desktop,
WSL2): os três containers sobem com um único `docker compose up --build`, o
Prometheus coleta a API (`up{job="triage-api"} = 1`) e o dashboard é
provisionado automaticamente no Grafana, exibindo dado real após tráfego no
`/triage`.

---

## CI/CD

`.github/workflows/ci.yml`: 3 jobs encadeados (`needs`) — **lint** (ruff) →
**test** (pytest, com o modelo treinado no próprio job) → **build**
(`docker build`, sem push). Os três jobs executam com sucesso no GitHub
Actions a cada push na `main` — ver a aba **Actions** do repositório.

---

## Orquestração (Airflow)

`airflow/dags/retrain_dag.py` — DAG `medical_triage_retrain` com 3 tasks via
TaskFlow API: `load_data → train_and_save_model → report_metrics`. A lógica
de cada task vive em `src/triage/` (testada por `pytest` independente do
Airflow); o arquivo da DAG só orquestra.

O Airflow **não** faz parte das dependências do projeto (`pyproject.toml`)
de propósito: ele carrega mais de cem pacotes transitivos com pins próprios,
o que engessaria o ambiente da API, e a versão 2.x não suporta Python 3.13+.
A DAG é executada em um ambiente separado, o que mantém `uv sync` leve e
rápido para quem só quer rodar a API ou os testes.

Para reproduzir a execução da DAG (requer Python 3.11 ou 3.12):

```bash
python -m venv .venv-airflow && source .venv-airflow/bin/activate
# no Windows: .venv-airflow\Scripts\activate

AIRFLOW_VERSION=2.10.5
PY_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
pip install "apache-airflow==${AIRFLOW_VERSION}" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PY_VERSION}.txt"
pip install -e .   # instala o pacote triage no mesmo venv

export AIRFLOW_HOME="$(pwd)/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$(pwd)/airflow/dags"
airflow dags test medical_triage_retrain 2026-01-01
```

Resultado obtido com Airflow 2.10.5: as 3 tasks concluíram com `SUCCESS`
(`macro_f1=0.6347`). O atalho `make dag-test` roda o último comando
assumindo que o venv acima já está ativo.

---

## Monitoramento

A API expõe `/metrics` (formato Prometheus) via `prometheus-client`, com:

- `triage_requests_total{endpoint, status_code}` — contagem de requisições.
- `triage_request_latency_seconds{endpoint}` — histograma de latência.
- `triage_predictions_total{urgency}` — predições por nível de urgência.
- `triage_errors_total{endpoint}` — contagem de erros.

Dashboard Grafana (`monitoring/grafana/dashboards/triage_api.json`), com
**4 painéis** (mínimo exigido: 3): total de requisições, taxa de erro,
latência p50/p95 e predições por nível de urgência. Datasource e dashboard
são provisionados automaticamente ao subir o `docker compose` — sem
configuração manual na UI do Grafana (o dashboard fica em **Dashboards →
Triagem de Laudos — API**).

O painel de taxa de erro usa `or vector(0)` na expressão para exibir 0% em
vez de "No data" enquanto nenhum erro tiver ocorrido — que é o estado
saudável esperado, e não ausência de métrica.

---

## Otimização de latência

Técnica aplicada: conversão do classificador para **ONNX Runtime** (via
`skl2onnx`) — ver justificativa de escopo e resultados completos em
[`reports/latency_comparison.md`](reports/latency_comparison.md).

| Métrica | Baseline (sklearn) | ONNX Runtime | Speedup |
|---|---|---|---|
| p50 | 0.73 ms | 0.64 ms | 1.14x |
| p95 | 0.90 ms | 0.82 ms | 1.10x |

A API pode servir por qualquer um dos dois backends via
`TRIAGE_USE_ONNX_RUNTIME=true` (variável de ambiente lida por
`triage.config.Settings`).

### Baseline de latência da API em Docker

Medido com a stack completa de pé, 200 requisições sequenciais ao `/triage`
(20 de aquecimento descartadas), a partir do host para o container:

| Métrica | Fim a fim (cliente → API) |
|---|---|
| p50 | 2.31 ms |
| p95 | 2.49 ms |
| média | 2.47 ms |

Esse número inclui a pilha HTTP e o overhead do cliente, portanto é o limite
superior honesto do tempo de resposta percebido. A latência medida **dentro**
da API (histograma `triage_request_latency_seconds`) aparece no painel
"Latência da API (p50 / p95)" do Grafana e é menor, por não contar o
round-trip — os dois juntos delimitam o intervalo real.

---

## Estrutura do repositório

```
medical-triage-mlops/
├── src/triage/
│   ├── config.py                 # Settings (Pydantic)
│   ├── data/{labels,loader}.py   # rótulo de urgência + carga do dataset
│   ├── models/
│   │   ├── train.py              # TF-IDF + LogisticRegression
│   │   ├── baselines.py          # majority-class + Naive Bayes (comparação)
│   │   ├── optimize.py           # export ONNX + benchmark de latência
│   │   └── predict.py            # TriagePredictor (wrapper de inferência)
│   └── api/main.py               # FastAPI + instrumentação Prometheus
├── airflow/dags/retrain_dag.py    # DAG de retreino
├── monitoring/                   # prometheus.yml + provisioning Grafana
├── tests/                        # 14 testes (pytest)
├── reports/                      # latency_comparison.{json,md}
├── data/raw/                     # Medical Abstracts TC Corpus (CC BY-SA 3.0)
├── Dockerfile                    # multi-stage (builder uv + runtime)
├── docker-compose.yml            # API + Prometheus + Grafana
├── .github/workflows/ci.yml      # lint -> test -> build
├── pyproject.toml / uv.lock
└── docs/model_card.md
```

---

## Resultados

- **Modelagem**: TF-IDF + Regressão Logística, macro-F1 = **0.6347** no
  teste (vs. 0.2025 da classe majoritária e 0.5374 de um Naive Bayes) —
  detalhes e limitações em [`docs/model_card.md`](docs/model_card.md).
- **CI/CD**: workflow com 2 automações (lint + test) além do build, com
  execução verde no GitHub Actions.
- **Orquestração**: DAG Airflow validada de ponta a ponta.
- **Monitoramento**: stack completa via um único `docker compose up`, com
  dashboard provisionado e p50/p95 da API a ~2.3-2.5 ms fim a fim.
- **Latência**: ONNX Runtime ~1.1-1.14x mais rápido que o baseline
  scikit-learn, com a limitação de escopo documentada.
