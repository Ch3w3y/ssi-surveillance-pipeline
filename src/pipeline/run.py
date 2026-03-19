"""End-to-end SSI surveillance pipeline orchestrator.

Coordinates preprocessing, NER, classification, and output formatting.
NER and BERT run sequentially per note for memory efficiency on NHS
workstation hardware (no batched GPU processing assumed).
"""
from __future__ import annotations
import pandas as pd
import yaml
from ..preprocessing.preprocessor import Preprocessor
from ..ner.pipeline import NERPipeline
from ..ner.assertion import format_entity_output
from ..classifier.model import ClinicalBERTClassifier
from ..classifier.structured import ICD10RuleEngine
from ..classifier.calibration import assign_confidence_zone, DEFAULT_THRESHOLDS
from ..output.formatter import format_linelist

TEXT_COLUMNS = [
    "note_text", "presenting_complaint", "clinical_findings",
    "diagnosis", "management_plan", "discharge_summary",
]


class SSIPipeline:
    """End-to-end SSI surveillance pipeline.

    Args:
        config: Pipeline configuration dict (loaded from config.yaml).
    """

    def __init__(self, config: dict):
        self.config = config
        self.preprocessor = Preprocessor(config)
        self.thresholds = config.get("thresholds", DEFAULT_THRESHOLDS)
        self._ner = None
        self._classifier = None
        self._structured = ICD10RuleEngine()

    @classmethod
    def from_config(cls, config_path: str) -> "SSIPipeline":
        """Instantiate from a YAML config file path."""
        with open(config_path) as f:
            return cls(yaml.safe_load(f))

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full pipeline on a batch DataFrame.

        Args:
            df: Raw input DataFrame (Format A or B).

        Returns:
            Full line list DataFrame with all output columns.
        """
        df = self.preprocessor.run(df)
        mode = self._detect_mode(df)
        results = []

        for _, row in df.iterrows():
            result_row = dict(row)
            result_row["processing_mode"] = mode

            if result_row.get("ssi_classification", "") not in ("", None):
                result_row.setdefault("extracted_entities", "")
                result_row.setdefault("entity_snippets", "")
                results.append(result_row)
                continue

            if mode == "structured_only":
                clf = self._structured.classify(row.get("icd10_codes"))
            else:
                clf = self._run_text_classification(row)

            result_row.update(clf)
            results.append(result_row)

        return format_linelist(pd.DataFrame(results))

    def _detect_mode(self, df: pd.DataFrame) -> str:
        configured = self.config.get("processing_mode", "auto")
        if configured != "auto":
            return configured
        return "text_only" if any(c in df.columns for c in TEXT_COLUMNS) else "structured_only"

    def _run_text_classification(self, row: pd.Series) -> dict:
        """Run NER + BERT classifier on a single note row."""
        if self._ner is None:
            self._ner = NERPipeline()
        if self._classifier is None:
            self._classifier = ClinicalBERTClassifier(
                self.config.get("model", "Simonlee711/Clinical_ModernBERT"),
                thresholds=self.thresholds,
            )
        text = str(row.get("note_text", ""))
        entities = self._ner.run(text)
        entity_pairs, entity_snippets = format_entity_output(entities)
        probs = self._classifier.classify(
            text=text,
            procedure_type=str(row.get("procedure_type", "")),
            days_post_op=row.get("days_post_op"),
            ecdc_window=str(row.get("ecdc_window_flag", "unknown")),
        )
        zone, review = assign_confidence_zone(probs, self.thresholds)
        if zone == "auto_positive":
            ssi_class = max((k for k in probs if k != "none"), key=lambda k: probs[k])
        else:
            ssi_class = "none"
        return {
            "ssi_classification": ssi_class,
            "p_none": probs["none"],
            "p_superficial": probs["superficial"],
            "p_deep": probs["deep"],
            "p_organ_space": probs["organ_space"],
            "confidence_zone": zone,
            "review_required": review,
            "extracted_entities": entity_pairs,
            "entity_snippets": entity_snippets,
        }
