"""Clinical_ModernBERT sequence classifier wrapper.

Wraps HuggingFace AutoModelForSequenceClassification with:
- Metadata token prepending (procedure type, days_post_op, ECDC window)
- Temperature-scaled probability output
- ECDC post-softmax gating

The model is downloaded from HuggingFace Hub on first instantiation (~400 MB).
Subsequent runs use the local HuggingFace cache.
"""
from __future__ import annotations
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .calibration import apply_temperature, DEFAULT_THRESHOLDS
from .ecdc_gating import apply_ecdc_gating

CLASS_ORDER = ["none", "superficial", "deep", "organ_space"]


class ClinicalBERTClassifier:
    """Transformer-based 4-class SSI classifier.

    Args:
        model_name: HuggingFace model identifier.
        temperature: Calibration temperature (1.0 = uncalibrated).
        thresholds: Triage threshold dict.
    """

    def __init__(self, model_name: str, temperature: float = 1.0, thresholds: dict | None = None):
        self.model_name = model_name
        self.temperature = temperature
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=4
        )
        self.model.train(False)  # set to inference mode

    def classify(
        self,
        text: str,
        procedure_type: str,
        days_post_op: int | None,
        ecdc_window: str,
    ) -> dict[str, float]:
        """Classify a single note and return calibrated, gated probabilities.

        Args:
            text: Cleaned note text.
            procedure_type: e.g. 'hip_total'.
            days_post_op: Integer days post-op, or None.
            ecdc_window: ECDC window label.

        Returns:
            Dict mapping class names to calibrated, ECDC-gated probabilities.
        """
        input_text = (
            f"[PROCEDURE: {procedure_type}] "
            f"[DAYS_POST_OP: {days_post_op if days_post_op is not None else 'unknown'}] "
            f"[WINDOW: {ecdc_window}] {text}"
        )
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=8192)
        import torch
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits.detach().numpy()[0]
        probs_array = apply_temperature(logits, self.temperature)
        probs = dict(zip(CLASS_ORDER, probs_array.tolist()))
        return apply_ecdc_gating(probs, days_post_op)
