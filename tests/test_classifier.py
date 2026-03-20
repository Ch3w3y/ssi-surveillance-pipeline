# tests/test_classifier.py
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_classifier():
    with patch(
        "src.classifier.model.AutoModelForSequenceClassification"
    ) as MockModel, patch("src.classifier.model.AutoTokenizer") as MockTokenizer:
        mock_out = MagicMock()
        mock_out.logits.detach.return_value.numpy.return_value = np.array(
            [[1.5, 0.5, 0.3, 0.1]]
        )
        MockModel.from_pretrained.return_value = MagicMock(return_value=mock_out)
        MockTokenizer.from_pretrained.return_value = MagicMock(
            return_value={"input_ids": MagicMock(), "attention_mask": MagicMock()}
        )
        from src.classifier.model import ClinicalBERTClassifier

        yield ClinicalBERTClassifier("Simonlee711/Clinical_ModernBERT")


def test_returns_four_probs(mock_classifier):
    result = mock_classifier.classify("Wound healing.", "hip_total", 10, "within_30d")
    assert set(result.keys()) >= {"none", "superficial", "deep", "organ_space"}


def test_probs_sum_to_one(mock_classifier):
    result = mock_classifier.classify("Wound healing.", "hip_total", 10, "within_30d")
    assert abs(sum(result.values()) - 1.0) < 1e-5


def test_metadata_in_tokeniser_input(mock_classifier):
    mock_classifier.classify("Some note.", "knee_total", 45, "within_1yr")
    call_args = mock_classifier.tokenizer.call_args
    text_arg = call_args[0][0] if call_args[0] else str(call_args)
    assert "knee_total" in text_arg or "PROCEDURE" in text_arg
