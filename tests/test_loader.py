import pandas as pd
import pytest

from triage.data.loader import add_urgency_label, load_raw_split


def test_load_raw_split_requires_expected_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError, match="Colunas ausentes"):
        load_raw_split(bad_csv)


def test_add_urgency_label_derives_column_and_strips_text():
    df = pd.DataFrame(
        {
            "condition_label": [1, 4],
            "medical_abstract": ["  texto A  ", "texto B"],
        }
    )
    out = add_urgency_label(df)
    assert list(out["urgency"]) == ["urgente", "urgente"]
    assert out["medical_abstract"].iloc[0] == "texto A"
