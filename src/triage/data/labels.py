"""Mapeamento das classes originais do dataset para níveis de urgência.

O Medical Abstracts TC Corpus (Schopf et al., 2022) rotula resumos de
literatura médica em 5 categorias de condição clínica — não é um dataset de
laudos de pronto-socorro com rótulo de urgência real. Para o cenário
fictício deste Tech Challenge ("sistema de triagem hospitalar"), agrupamos
as 5 categorias em 3 níveis de urgência (normal / atenção / urgente).

⚠️ Isto é uma simplificação didática para exercitar o pipeline de MLOps
(deploy, CI/CD, monitoramento, otimização de latência) — **não** é uma regra
clínica validada e não deve ser usada para triagem real. Ver docs/model_card.md
para a discussão completa da limitação.

Critério adotado (documentado, não clínico):
- "urgente": categorias tipicamente associadas a quadros agudos/sistêmicos
  na literatura (neoplasias, doenças cardiovasculares).
- "atenção": categorias de evolução mais variável, nem sempre emergencial
  (doenças digestivas, condições patológicas gerais).
- "normal": doenças do sistema nervoso, tratada aqui como classe de
  referência de menor prioridade imediata dentro deste recorte didático.
"""

from enum import IntEnum


class ConditionLabel(IntEnum):
    """Rótulo original do corpus (1-5)."""

    NEOPLASMS = 1
    DIGESTIVE = 2
    NERVOUS = 3
    CARDIOVASCULAR = 4
    GENERAL_PATHOLOGICAL = 5


CONDITION_NAMES: dict[int, str] = {
    ConditionLabel.NEOPLASMS: "neoplasms",
    ConditionLabel.DIGESTIVE: "digestive system diseases",
    ConditionLabel.NERVOUS: "nervous system diseases",
    ConditionLabel.CARDIOVASCULAR: "cardiovascular diseases",
    ConditionLabel.GENERAL_PATHOLOGICAL: "general pathological conditions",
}

URGENCY_LEVELS: tuple[str, ...] = ("normal", "atencao", "urgente")

_CONDITION_TO_URGENCY: dict[int, str] = {
    ConditionLabel.NEOPLASMS: "urgente",
    ConditionLabel.CARDIOVASCULAR: "urgente",
    ConditionLabel.DIGESTIVE: "atencao",
    ConditionLabel.GENERAL_PATHOLOGICAL: "atencao",
    ConditionLabel.NERVOUS: "normal",
}


def condition_to_urgency(condition_label: int) -> str:
    """Converte o rótulo original (1-5) no nível de urgência fictício."""
    try:
        return _CONDITION_TO_URGENCY[ConditionLabel(condition_label)]
    except ValueError as exc:
        raise ValueError(f"condition_label inválido: {condition_label}") from exc


def urgency_to_index(urgency: str) -> int:
    """Índice estável (0..2) de um nível de urgência, para métricas/matrizes."""
    return URGENCY_LEVELS.index(urgency)
