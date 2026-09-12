from triage.data.labels import URGENCY_LEVELS, condition_to_urgency, urgency_to_index


def test_condition_to_urgency_maps_all_five_classes():
    mapped = {condition_to_urgency(i) for i in range(1, 6)}
    assert mapped == set(URGENCY_LEVELS)


def test_condition_to_urgency_is_deterministic():
    assert condition_to_urgency(1) == condition_to_urgency(1)
    assert condition_to_urgency(4) == "urgente"
    assert condition_to_urgency(3) == "normal"


def test_condition_to_urgency_rejects_invalid_label():
    import pytest

    with pytest.raises(ValueError):
        condition_to_urgency(99)


def test_urgency_to_index_stable_order():
    assert urgency_to_index("normal") == 0
    assert urgency_to_index("urgente") == len(URGENCY_LEVELS) - 1
