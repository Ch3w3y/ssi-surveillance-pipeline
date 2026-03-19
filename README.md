# bert_SSI

![CI](https://github.com/your-org/bert_SSI/actions/workflows/ci.yml/badge.svg)
![Coverage](https://codecov.io/gh/your-org/bert_SSI/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)
![Licence](https://img.shields.io/badge/licence-Apache%202.0-green)

Automated surveillance of Surgical Site Infections (SSI) in orthopaedic hip and knee arthroplasty using transformer-based NLP on post-surgical clinical notes. Implements ECDC HAI-Net SSI definitions with a human-in-the-loop review workflow designed for NHS epidemiological surveillance.

> **This tool is for epidemiological surveillance only. It is not validated for clinical decision-making or individual patient management.**

---

## Contents

- [Overview](#overview)
- [ECDC SSI Definitions](#ecdc-ssi-definitions)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Input Data](#input-data)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Output](#output)
- [Operating Modes](#operating-modes)
- [Training Your Own Model](#training-your-own-model)
- [Evaluation](#evaluation)
- [NER Entity Schema](#ner-entity-schema)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [Licence](#licence)
- [Citation](#citation)
- [References](#references)

---

## Overview

`bert_SSI` classifies post-surgical consultation notes to detect the presence and type of SSI according to ECDC criteria. For each episode it outputs:

- A 4-class ECDC label: **none / superficial incisional / deep incisional / organ-space**
- Calibrated probability scores for each class
- Extracted NER entity spans with assertion status (affirmed / negated / uncertain) — showing *which* text drove the classification
- A triage zone: **auto-negative**, **review-required**, or **auto-positive**

Cases in the review-required zone are written to a separate MDT review list. This human-in-the-loop design means clinical teams review only a small proportion of episodes (~3–5% in published comparable systems), rather than manually auditing every record.

The pipeline operates in two modes:

| Mode | Input | Use case |
|---|---|---|
| `text_only` | Free text clinical notes | Primary mode — highest sensitivity |
| `structured_only` | ICD-10 codes only | Fallback for administrative-data-only datasets |

Comparing performance between these two modes quantifies the surveillance gap attributable to lack of free text linkage — a key evidence base for stakeholders evaluating the investment in linking surgical systems to administrative datasets.

---

## ECDC SSI Definitions

Classifications follow **ECDC HAI-Net SSI Protocol v2.2**. All in-scope procedures (hip/knee arthroplasty) involve prosthetic implants, so the surveillance window for deep incisional and organ/space SSI extends to **1 year post-operatively**.

| SSI Type | Tissue Depth | Surveillance Window |
|---|---|---|
| Superficial Incisional | Skin and subcutaneous tissue | 30 days |
| Deep Incisional | Fascia and muscle | 1 year (implant) |
| Organ/Space | Beyond body wall (joint, periprosthetic) | 1 year (implant) |

When criteria are met at multiple levels, only the deepest level is reported (ECDC hierarchy). `days_post_op` is computed automatically from `operation_date` and `note_date` and gates which ECDC classes are clinically possible — ensuring classifications are always temporally valid regardless of model output.

---

## Quick Start

```bash
# Install
git clone https://github.com/your-org/bert_SSI.git
cd bert_SSI
pip install -e ".[dev]"
python -m spacy download en_core_sci_sm  # scispaCy base model

# Run the pipeline on a CSV of clinical notes
python scripts/run_pipeline.py --input data/my_notes.csv --output results/

# Output files:
#   results/ssi_linelist_YYYYMMDD.csv   — full episode-level results
#   results/ssi_review_YYYYMMDD.csv     — borderline cases for MDT review
#   results/ssi_summary_YYYYMMDD.txt    — surveillance summary
```

---

## Installation

**Requirements:** Python 3.9–3.11, 8 GB RAM minimum, CPU inference supported.

```bash
git clone https://github.com/your-org/bert_SSI.git
cd bert_SSI
pip install -e .
```

For development (includes linting and test dependencies):

```bash
pip install -e ".[dev]"
```

Install the scispaCy language model (required for NER):

```bash
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz
```

The classifier backbone (`Simonlee711/Clinical_ModernBERT`) is downloaded automatically from HuggingFace on first run (~400 MB). An internet connection is required for the first run; subsequent runs use the cached model.

**Estimated throughput (CPU, Intel Core i7, 16 GB RAM):**

| Hardware | Notes per minute | 500 episodes |
|---|---|---|
| CPU (standard workstation) | 8–15 | ~35–60 minutes |
| GPU (CUDA) | 100–200 | ~3–5 minutes |

---

## Input Data

The pipeline accepts a CSV file. Two input formats are supported and auto-detected from column names.

### Format A — Single text column

Minimum required columns:

| Column | Type | Description |
|---|---|---|
| `patient_id` | string | Pseudonymised identifier |
| `episode_id` | string | **Required** — used by MDT to retrieve case from clinical systems |
| `operation_date` | YYYY-MM-DD | Date of index surgical procedure |
| `note_date` | YYYY-MM-DD | Date of the clinical note |
| `procedure_code` | OPCS-4 | See [Supported Procedure Codes](#supported-procedure-codes) |
| `note_text` | string | Full free text of the clinical note |

Optional columns:

| Column | Type | Description |
|---|---|---|
| `icd10_codes` | pipe-separated | e.g. `Z96.6\|M16.1` — reserved for future hybrid mode |
| `hospital_site` | string | Site identifier for multi-site stratification |

### Format B — Multiple text columns (administrative data style)

Replace `note_text` with any combination of:

| Column | Description |
|---|---|
| `presenting_complaint` | Chief complaint or reason for attendance |
| `clinical_findings` | Examination findings, wound assessment |
| `diagnosis` | Stated diagnosis or impression |
| `management_plan` | Treatment plan, follow-up instructions |
| `discharge_summary` | Discharge narrative |

At least one text column must be non-null per row. The pipeline concatenates them in the order defined in `config.yaml`, inserting section headers between blocks.

### `episode_id` requirement

`episode_id` is **required**. The MDT review list uses it to identify cases for retrieval from clinical systems. The pipeline will halt with a validation error if `episode_id` is absent from the input.

### `structured_only` mode — ICD-10 only

If no text columns are present but `icd10_codes` is present, the pipeline automatically runs in `structured_only` mode using a deterministic ICD-10 rule engine. No text column is required. See [Operating Modes](#operating-modes).

### Supported Procedure Codes

In-scope OPCS-4 procedure codes (all involve prosthetic implants; 1-year surveillance window applies):

| Code | Description | Type |
|---|---|---|
| W37 | Total prosthetic replacement of hip joint NEC | Hip total |
| W38 | Total prosthetic replacement of hip joint using cement | Hip total |
| W39 | Prosthetic replacement of head of femur NEC | Hip hemi |
| W40 | Prosthetic replacement of head of femur using cement | Hip hemi |
| W41 | Revision of prosthetic replacement of hip joint NEC | Hip revision |
| W42 | Revision of prosthetic replacement of hip joint using cement | Hip revision |
| W43 | Primary total prosthetic replacement of knee joint NEC | Knee total |
| W44 | Primary total prosthetic replacement of knee joint using cement | Knee total |
| W45 | Revision of prosthetic replacement of knee joint NEC | Knee revision |
| W46 | Revision of prosthetic replacement of knee joint using cement | Knee revision |
| W47 | Other prosthetic replacement of knee joint | Knee other |

Episodes with codes outside this list are flagged `out_of_scope` in the output rather than silently discarded.

---

## Configuration

All parameters are set in `config.yaml`:

```yaml
# Processing
processing_mode: auto          # auto | text_only | structured_only | hybrid
input_format: auto             # auto | A | B

# Text column concatenation order (Format B)
text_columns:
  - field: presenting_complaint
    header: "PRESENTING COMPLAINT"
  - field: clinical_findings
    header: "CLINICAL FINDINGS"
  - field: diagnosis
    header: "DIAGNOSIS"
  - field: management_plan
    header: "MANAGEMENT PLAN"
  - field: discharge_summary
    header: "DISCHARGE SUMMARY"

# Classifier
model: Simonlee711/Clinical_ModernBERT
comparison_model: emilyalsentzer/Bio_ClinicalBERT  # for ablation evaluation only

# Triage thresholds (adjustable for local sensitivity/specificity trade-off)
thresholds:
  auto_negative: 0.85          # P(none) ≥ this → auto-negative
  auto_positive: 0.85          # max(SSI classes) ≥ this → auto-positive
  # All other rows → review_required (catch-all)

# Output
output_dir: results/
```

Surveillance teams can adjust `auto_negative` and `auto_positive` thresholds to tune the sensitivity/specificity trade-off for their local context without modifying code.

---

## Running the Pipeline

### Batch run (standard usage)

```bash
python scripts/run_pipeline.py \
  --input data/my_notes.csv \
  --output results/
```

### Specify processing mode explicitly

```bash
# Force structured-only mode (ICD-10 codes only, no text required)
python scripts/run_pipeline.py \
  --input data/pedw_extract.csv \
  --output results/ \
  --mode structured_only

# Force text-only mode
python scripts/run_pipeline.py \
  --input data/clinic_letters.csv \
  --output results/ \
  --mode text_only
```

### Run via Python

```python
from src.pipeline.run import SSIPipeline

pipeline = SSIPipeline.from_config("config.yaml")
results = pipeline.run("data/my_notes.csv")
results.to_csv("results/ssi_linelist.csv", index=False)
```

---

## Output

Every run produces three files in the output directory, named with a YYYYMMDD timestamp.

### `ssi_linelist_YYYYMMDD.csv` — full results

One row per input episode. Key columns:

| Column | Description |
|---|---|
| `patient_id`, `episode_id` | From input |
| `days_post_op` | Computed from operation and note dates |
| `procedure_type` | Derived: `hip_total` / `hip_hemi` / `hip_revision` / `knee_total` / `knee_revision` |
| `processing_mode` | `text_only` / `structured_only` / `hybrid` |
| `ssi_classification` | `none` / `superficial` / `deep` / `organ_space` / flag (see below) |
| `p_none`, `p_superficial`, `p_deep`, `p_organ_space` | Calibrated probabilities (sum to 1.0) |
| `confidence_zone` | `auto_negative` / `review_required` / `auto_positive` |
| `review_required` | Boolean |
| `extracted_entities` | Pipe-separated `ENTITY_TYPE:assertion` pairs |
| `entity_snippets` | Pipe-separated text spans that triggered each entity |
| `ecdc_window_flag` | `within_30d` / `within_1yr` / `outside_window` |

**Flag values for `ssi_classification`** (rows excluded from SSI classification):

| Flag | Meaning |
|---|---|
| `out_of_scope` | Procedure code not in W37–W47 |
| `outside_window` | Note date > 365 days post-op |
| `missing_operation_date` | `operation_date` null |
| `missing_note_date` | `note_date` null |
| `invalid_dates` | `note_date` before `operation_date` |
| `insufficient_data` | All text columns null (text modes) |

No rows are silently discarded — all inputs appear in the output with a status.

### `ssi_review_YYYYMMDD.csv` — MDT review list

Filtered to `review_required = True`, sorted by maximum SSI class probability descending. Includes a blank `reviewer_notes` column for clinical team use.

### `ssi_summary_YYYYMMDD.txt` — run summary

```
SSI Surveillance Pipeline — Run Summary
========================================
Run date           : 2026-03-19
Processing mode    : text_only
Episodes processed : 1,847
  Out of scope     :    23
  Missing dates    :     4
  Insufficient data:    11

Classifications (ECDC):
  None             : 1,764  (97.0%)
  Superficial SSI  :    21  (1.16%)
  Deep SSI         :    14  (0.77%)
  Organ/Space SSI  :     6  (0.33%)
  Overall SSI rate :  2.26% (95% CI: 1.67–2.97%)

Review-required (borderline): 48 episodes (2.6%)

Thresholds applied:
  auto_negative : P(none) ≥ 0.85
  auto_positive : P(SSI)  ≥ 0.85
```

---

## Operating Modes

### `text_only` (recommended)

Uses the full NER + transformer classifier pipeline on free text. Highest sensitivity and specificity. Requires at least one text column per row.

### `structured_only`

Uses a deterministic rule engine on ICD-10 codes only. No text required. Replicates the capability of existing administrative-data-based surveillance.

**Expected performance gap:** Published literature reports sensitivity ~10% for ICD-10 coded data vs ~85%+ for free text NLP. Notebook `notebooks/06_structured_vs_text.ipynb` reproduces this comparison on the evaluation dataset, providing evidence for stakeholder decisions about free text data linkage.

Relevant NHS ICD-10 codes used in this mode:

| Code | Signal |
|---|---|
| T81.4 | Infection following procedure |
| T84.5 | Infection/inflammatory reaction due to internal joint prosthesis |
| T84.6 | Infection due to internal fixation device |
| L02, L03 | Superficial wound infection / cellulitis |
| M00.8, M00.9 | Pyogenic arthritis (organ-space proxy) |

> Note: These are **NHS ICD-10 (WHO)** codes, not ICD-10-CM. They differ from the US coding system in their level of subdivision.

---

## Training Your Own Model

The pre-trained model weights are available on HuggingFace (see [Citation](#citation)). To fine-tune on your own data:

### 1. Obtain MIMIC-IV-Note access

Training uses silver-labelled data derived from MIMIC-IV-Note (PhysioNet). Access requires:
1. Register at [physionet.org](https://physionet.org)
2. Complete CITI "Data or Specimens Only Research" training
3. Sign the MIMIC-IV-Note data use agreement
4. Download MIMIC-IV-Note v2.2

Full instructions: `training/README.md`

### 2. Build the training cohort

```bash
python training/mimic_silver_labels.py \
  --mimic_dir /path/to/mimic-iv-note \
  --output data/processed/silver_labels.csv
```

This filters MIMIC-IV by orthopaedic procedure codes and SSI-related ICD-10-CM codes, links notes, and applies ECDC sub-type heuristics to produce silver labels.

> **Note on coding systems:** MIMIC-IV uses ICD-10-CM (US). The training pipeline uses ICD-10-CM codes (`T84.50`–`T84.54`). The `structured_only` inference mode uses NHS ICD-10 (`T84.5`). These are documented separately and handled explicitly in the codebase.

### 3. Fine-tune

```bash
python training/train.py \
  --data data/processed/silver_labels.csv \
  --output models/classifier/ \
  --model Simonlee711/Clinical_ModernBERT
```

### 4. Evaluate against gold-standard annotations

See `data/annotations/annotation_guide.md` for the ECDC-based annotation protocol. Inter-annotator agreement threshold: Cohen's kappa ≥ 0.70.

```bash
python training/evaluate.py \
  --model models/classifier/ \
  --annotations data/annotations/gold_standard.csv
```

---

## Evaluation

Performance is reported against a manually annotated gold-standard evaluation set following ECDC SSI criteria. Metrics are surveillance-oriented:

| Metric | Rationale |
|---|---|
| Sensitivity (Recall) | Missed cases undercount true SSI burden |
| Specificity | Controls false positive rate and MDT workload |
| NPV | Validates the auto-negative zone |
| AUC-ROC | Overall discrimination across all thresholds |
| Calibration (Brier score) | Validates that probability scores are meaningful |

Subgroup analyses are reported by ECDC SSI type, procedure type (hip total / hip hemi / hip revision / knee total / knee revision), days post-op band, and processing mode.

Evaluation notebooks: `notebooks/05_evaluation.ipynb` and `notebooks/06_structured_vs_text.ipynb`.

---

## NER Entity Schema

The NER layer extracts ten entity types from free text. Each entity is tagged with an assertion status by the ConText algorithm.

| Entity Type | Examples |
|---|---|
| `WOUND_SIGN` | redness, erythema, swelling, warmth, tenderness |
| `DISCHARGE` | purulent discharge, pus, seropurulent, exudate |
| `WOUND_DISRUPTION` | dehiscence, wound breakdown, wound opened |
| `ABSCESS` | abscess, collection, fluid collection |
| `FEVER` | fever, pyrexia, temperature 38.5, febrile |
| `ANTIBIOTIC` | co-amoxiclav, flucloxacillin, IV antibiotics |
| `WOUND_TREATMENT` | debridement, washout, reoperation, DAIR |
| `MICROBIOLOGY` | wound swab, culture positive, MRSA, Staph aureus |
| `ANATOMICAL_DEPTH` | superficial, deep, fascia, joint, prosthesis |
| `TEMPORAL` | post-op day 5, 3 weeks post surgery |

**Assertion status** (via ConText):

| Status | Example |
|---|---|
| `affirmed` | *"purulent discharge noted"* |
| `negated` | *"no signs of infection"*, *"antibiotics not required"* |
| `uncertain` | *"? early wound breakdown"* |
| `hypothetical` | *"if signs of infection develop"* |
| `historical` | *"previously had wound infection"* |

Assertion status is critical: *"antibiotics not required"* produces `ANTIBIOTIC:negated` — a negative signal — rather than incorrectly contributing to a higher SSI probability. Entity rules are defined in `src/ner/entity_rules.py` and can be extended by clinical teams.

---

## Repository Structure

```
bert_SSI/
├── README.md
├── LICENSE
├── config.yaml
├── requirements.txt
├── setup.py
├── data/
│   ├── annotations/
│   │   ├── annotation_guide.md    # ECDC annotation protocol
│   │   └── schema.json            # Entity schema definition
│   └── reference/
│       ├── opcs4_orthopaedic.csv  # OPCS-4 reference
│       └── icd10_ssi_codes.csv    # NHS ICD-10 SSI codes
├── src/
│   ├── preprocessing/             # Text cleaning, temporal features, validation
│   ├── ner/                       # MedSpaCy + ConText pipeline and entity rules
│   ├── classifier/                # Clinical_ModernBERT fine-tuning and inference
│   ├── pipeline/                  # End-to-end orchestration
│   └── output/                    # Line list and summary generation
├── training/
│   ├── README.md                  # MIMIC-IV access and cohort instructions
│   ├── mimic_silver_labels.py
│   ├── ecdc_heuristics.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_silver_label_analysis.ipynb
│   ├── 03_ner_development.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_structured_vs_text.ipynb  # Key: text vs ICD-10 performance comparison
├── scripts/
│   ├── run_pipeline.py
│   └── annotate.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_ner.py
│   ├── test_classifier.py
│   ├── test_structured.py
│   ├── test_output.py
│   ├── test_pipeline.py
│   └── smoke/
│       ├── test_smoke_text_only.py
│       ├── test_smoke_structured_only.py
│       └── fixtures/synthetic_notes.csv
└── .github/
    └── workflows/
        └── ci.yml
```

---

## Contributing

Contributions are welcome — particularly:

- **Clinical entity rules** (`src/ner/entity_rules.py`): additional orthopaedic terminology, NHS trust-specific dictation patterns, new procedure vocabularies
- **Procedure scope extensions**: additional OPCS-4 codes beyond the current W37–W47 scope
- **Evaluation datasets**: annotated corpora from NHS settings
- **Bug reports and performance findings**

Please open an issue before submitting a PR for significant changes. See `data/annotations/annotation_guide.md` if contributing annotated data.

All contributions must include tests. The CI pipeline must pass before merging.

```bash
# Run tests locally before submitting
pytest tests/ -v
pytest tests/smoke/ -v
flake8 src/ tests/ --max-line-length=100
black --check src/ tests/
```

---

## Licence

Apache 2.0 — see [LICENSE](LICENSE).

This licence is compatible with NHS and public sector reuse and permits commercial adaptation.

**Training data:** MIMIC-IV-Note is subject to the [PhysioNet Credentialed Health Data Licence](https://physionet.org/content/mimiciii/view-license/1.4/). No MIMIC data is included in this repository. Users must obtain their own PhysioNet access.

**Pre-trained models:** Clinical_ModernBERT and Bio_ClinicalBERT are both MIT licenced and available via HuggingFace.

---

## Citation

If you use this tool in research, please cite:

```bibtex
@article{bert_ssi_2026,
  title   = {Automated Surveillance of Surgical Site Infections Using Transformer-based
             Natural Language Processing: A Reproducible Pipeline for NHS Administrative
             and Free-text Data},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {2026},
  doi     = {[DOI]}
}
```

---

## References

- van Mourik MSM et al. PRAISE: providing a roadmap for automated infection surveillance in Europe. *Clin Microbiol Infect.* 2021;27(S1):S3–S19. doi:10.1016/j.cmi.2021.02.028
- PRAISE SSI Working Group. Automated surveillance for surgical site infections — expert perspectives for implementation. *Antimicrob Resist Infect Control.* 2024. PMC11667888.
- Bucher et al. Portable Automated Surveillance of Surgical Site Infections Using NLP. *Ann Surg.* 2020. PMC9040555.
- Danish DL study. Assessing the utility of deep neural networks in detecting superficial SSI from free text EHR data. PMC10801170.
- Alsentzer et al. Publicly Available Clinical BERT Embeddings. NAACL 2019.
- Lee S. Clinical_ModernBERT. HuggingFace. 2025.
- ECDC HAI-Net SSI Protocol v2.2. European Centre for Disease Prevention and Control.
- UKHSA Protocol for the Surveillance of Surgical Site Infection. UK Health Security Agency.
