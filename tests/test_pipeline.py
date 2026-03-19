# tests/test_pipeline.py
import pandas as pd
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

MOCK_LOGITS = np.array([[0.8, 0.1, 0.06, 0.04]])


@pytest.fixture
def pipeline_with_mock():
    with patch("src.classifier.model.AutoModelForSequenceClassification") as MockModel, \
         patch("src.classifier.model.AutoTokenizer") as MockTok:
        mock_out = MagicMock()
        mock_out.logits.detach.return_value.numpy.return_value = MOCK_LOGITS
        MockModel.from_pretrained.return_value = MagicMock(return_value=mock_out)
        MockTok.from_pretrained.return_value = MagicMock(
            return_value={"input_ids": MagicMock(), "attention_mask": MagicMock()}
        )
        from src.pipeline.run import SSIPipeline
        config = {
            "model": "Simonlee711/Clinical_ModernBERT",
            "processing_mode": "auto",
            "text_columns": [],
            "thresholds": {"auto_negative": 0.85, "auto_positive": 0.85},
        }
        yield SSIPipeline(config)


def test_returns_dataframe(pipeline_with_mock, synthetic_df):
    assert isinstance(pipeline_with_mock.run(synthetic_df), pd.DataFrame)


def test_all_episodes_in_output(pipeline_with_mock, synthetic_df):
    result = pipeline_with_mock.run(synthetic_df)
    assert len(result) == len(synthetic_df)


def test_out_of_scope_flagged(pipeline_with_mock, synthetic_df):
    result = pipeline_with_mock.run(synthetic_df)
    oos = result[result["episode_id"] == "E003"]
    assert oos.iloc[0]["ssi_classification"] == "out_of_scope"


def test_required_output_columns(pipeline_with_mock, synthetic_df):
    result = pipeline_with_mock.run(synthetic_df)
    for col in ["ssi_classification", "p_none", "confidence_zone", "review_required"]:
        assert col in result.columns


def test_structured_only_mode(synthetic_df):
    from src.pipeline.run import SSIPipeline
    df = synthetic_df.drop(columns=["note_text"])
    df = df.copy()
    df["icd10_codes"] = "T81.4|Z96.6"
    config = {
        "processing_mode": "structured_only",
        "text_columns": [],
        "thresholds": {"auto_negative": 0.85, "auto_positive": 0.85},
    }
    result = SSIPipeline(config).run(df)
    assert isinstance(result, pd.DataFrame) and len(result) == len(df)
