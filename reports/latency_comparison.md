# Comparativo de Latência — Baseline vs. ONNX Runtime

Gerado por `python -m triage.models.optimize` (1000 requisições sequenciais,
20 chamadas de aquecimento descartadas antes da medição). Números completos
em `latency_comparison.json`; reprodutível a qualquer momento rodando o
mesmo comando.

| Métrica | Baseline (scikit-learn) | Otimizado (ONNX Runtime) | Speedup |
|---|---|---|---|
| p50 | 0.73 ms | 0.64 ms | **1.14x** |
| p95 | 0.90 ms | 0.82 ms | **1.10x** |
| p99 | 1.02 ms | 0.92 ms | 1.11x |
| média | 0.74 ms | 0.65 ms | 1.14x |

## Leitura honesta do resultado

O ganho é real, porém modesto (~10-14%), e isso é esperado dado o escopo da
otimização: convertemos para ONNX apenas o estágio de **classificação**
(`LogisticRegression`), não a vetorização TF-IDF, que continua em
scikit-learn (ver `src/triage/models/optimize.py` para a justificativa).
Para este modelo leve (20k features, 3 classes), a multiplicação
matriz-vetor do classificador já é rápida em NumPy puro — o ganho do
runtime ONNX vem de evitar a sobrecarga do dispatch Python/scikit-learn por
chamada, não de uma operação matematicamente mais pesada.

Em um cenário com um modelo maior (mais classes, embeddings densos, uma
rede neural) ou com inferência em lote, o ganho relativo do ONNX Runtime
tende a ser maior. Documentamos isso explicitamente em vez de inflar o
resultado — coerente com a decisão de manter o modelo intencionalmente leve
(ver README, seção "Decisão Arquitetural").
