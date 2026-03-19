# Design Specification: BERT-based SSI Surveillance Pipeline

**Date:** 2026-03-19
**Project:** bert_SSI
**Status:** Approved
**Scope:** Orthopaedic hip and knee replacement procedures (OPCS-4 W37–W47), UK NHS context
**Purpose:** Epidemiological surveillance of Surgical Site Infections (SSI) — not clinical management

---

## 1. Objectives

Develop a free text–first NLP pipeline to classify post-surgical consultation notes according to ECDC SSI definitions, producing calibrated probability scores and a stratified line list suitable for MDT review. The pipeline also operates in a structured-data-only mode using ICD-10 codes alone, enabling direct comparison of detection performance between coded administrative data and free text — a key stakeholder argument for investing in free text data linkage from surgical systems to administrative datasets.

The system targets:
- Epidemiological surveillance of hip and knee arthroplasty procedures in UK NHS settings
- Alignment with ECDC HAI-Net SSI Protocol v2.2 and UKHSA SSISS definitions
- Open release as a reproducible research tool for Public Health / NHS reuse

---

## 2. Background and Motivation

Automated SSI surveillance in the UK relies primarily on coded administrative data (ICD-10 via NHS administrative datasets such as HES or PEDW) and manual chart review. International evidence (PRAISE network, van Mourik et al. 2021; Danish DL study, PMC10801170) demonstrates that:

- ICD-10 coded data alone detects SSI with sensitivity ~10% (AUC ~0.55)
- Free-text NLP achieves sensitivity ~85%+ (AUC ~0.99) at scale
- A human-in-the-loop (HITL) workflow requiring MDT review of only ~3% of flagged cases approaches manual curation accuracy

UK NHS administrative datasets (HES in England, PEDW in Wales, SMR in Scotland) record ICD-10 coded diagnoses and OPCS-4 procedure codes alongside episode-level metadata. Free text from surgical systems (clinic letters, discharge summaries, wound assessment notes) is not yet routinely linked to these administrative datasets at scale. This pipeline is designed to:

1. Work immediately with any free text from NHS surgical systems
2. Fall back to structured-data-only mode against administrative data where no text is available
3. Quantify the performance gap between modes, making the evidence case for free text linkage

---

## 3. ECDC SSI Classification Framework

All classifications follow ECDC HAI-Net SSI Protocol v2.2. The surveillance window is procedure-dependent:

- **Superficial Incisional SSI:** within 30 days of procedure (all procedure types)
- **Deep Incisional SSI:** within 30 days (no implant) or **1 year** (implant in situ)
- **Organ/Space SSI:** within 30 days (no implant) or **1 year** (implant in situ)

Since all in-scope procedures (hip/knee arthroplasty) involve implants, deep incisional and organ/space SSI surveillance windows extend to **1 year post-operatively**. `days_post_op` is computed from `operation_date` and `note_date` and gates which ECDC classes are clinically possible prior to model scoring.

### 3.1 SSI Type Definitions

**Superficial Incisional SSI** — involves skin and subcutaneous tissue only. Requires ≥1 of:
- Purulent drainage from superficial incision
- Culture-positive fluid/tissue from superficial incision
- ≥1 symptom (pain, tenderness, swelling, redness, heat) AND deliberate opening of incision
- Surgeon/physician diagnosis

**Deep Incisional SSI** — involves deep soft tissues (fascia, muscle). Requires ≥1 of:
- Purulent drainage from deep incision (not from organ/space)
- Spontaneous dehiscence or deliberate opening with ≥1 sign of infection (fever >38°C, pain/tenderness)
- Abscess or deep infection found on examination, reoperation, histopathology, or radiology
- Surgeon/physician diagnosis

**Organ/Space SSI** — involves any anatomy beyond the body wall. Requires ≥1 of:
- Purulent drainage from a drain placed into the organ/space
- Culture-positive fluid/tissue from organ/space
- Abscess or infection found on examination, reoperation, histopathology, or radiology
- Surgeon/physician diagnosis

**Reporting rule:** When criteria are met at multiple levels, report only the deepest level (ECDC hierarchy).

---

## 4. Architecture

### 4.1 Overview

The pipeline has four sequential stages. NER and classification run in parallel at inference time and are merged by the output formatter.

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT LAYER                                                    │
│  CSV batch of post-surgical notes + metadata                    │
│  (patient_id, operation_date, note_date, procedure_code, text)  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  PREPROCESSOR                                                   │
│  • Input validation and bad-row flagging                        │
│  • Text cleaning (NHS encoding artefacts, whitespace)           │
│  • Multi-column text concatenation (Mode 2)                     │
│  • days_post_op computation                                     │
│  • ECDC window gating                                           │
│  • Section detection (impression, examination, plan)            │
└──────────────┬──────────────────────────┬───────────────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────┐    ┌─────────────────────────────────────┐
│  NER LAYER           │    │  CLASSIFIER                         │
│  MedSpaCy + ConText  │    │  Clinical_ModernBERT (fine-tuned)   │
│  Entity extraction   │    │  4-class: none / superficial /      │
│  + assertion status  │    │  deep / organ_space                 │
│  (affirmed/negated/  │    │  Calibrated probability scores      │
│   uncertain/etc.)    │    │  ECDC post-softmax gating           │
└──────────────┬───────┘    └─────────────────┬───────────────────┘
               │                              │
               └──────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT FORMATTER                                               │
│  • Merge NER spans + classifier decision + probabilities        │
│  • Apply threshold logic (auto / borderline / review-required)  │
│  • Emit CSV line list, MDT review list, run summary             │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Operating Modes

The pipeline auto-detects the operating mode from the columns present in the input CSV, or it can be set explicitly in `config.yaml`.

| Mode | Inputs Required | Components Active |
|---|---|---|
| `text_only` | Free text + dates | Preprocessor → NER + classifier → Output |
| `structured_only` | ICD-10 codes + dates | Preprocessor → ICD-10 rule engine → Output |
| `hybrid` *(future)* | Both | All components; structured features prepended as tokens |

The `structured_only` mode deliberately replicates the capability of existing administrative data-based surveillance, providing a direct performance comparator against `text_only` mode.

---

## 5. NER Layer

### 5.1 Framework

`MedSpaCy` with the `ConText` algorithm for assertion detection. Extended with custom orthopaedic entity rules covering NHS UK spelling variants, procedure-specific vocabulary, and common dictation artefacts.

### 5.2 Entity Schema

| Entity Type | Examples | Clinical Purpose |
|---|---|---|
| `WOUND_SIGN` | redness, erythema, swelling, warmth, tenderness, heat | Superficial SSI indicators |
| `DISCHARGE` | purulent discharge, pus, seropurulent, exudate, weeping | Primary ECDC criterion (all types) |
| `WOUND_DISRUPTION` | dehiscence, wound breakdown, wound opened, gaping | Superficial / deep indicator |
| `ABSCESS` | abscess, collection, fluid collection, haematoma infected | Deep / organ-space indicator |
| `FEVER` | fever, pyrexia, temperature 38.5, high temperature, febrile | Systemic infection marker |
| `ANTIBIOTIC` | co-amoxiclav, flucloxacillin, IV antibiotics, antimicrobial | Treatment proxy — context-dependent |
| `WOUND_TREATMENT` | debridement, washout, reoperation, wound exploration, DAIR | Deep / organ-space trigger |
| `MICROBIOLOGY` | wound swab, culture positive, MRSA, Staph aureus, organism isolated | ECDC criterion (any type) |
| `ANATOMICAL_DEPTH` | superficial, deep, fascia, muscle, joint, prosthesis, periprosthetic | SSI type discriminator |
| `TEMPORAL` | post-op day 5, 3 weeks post surgery, 8 months later | ECDC window validation |

### 5.3 Assertion Status

Each extracted entity span carries an assertion modifier determined by ConText:

| Status | Trigger Pattern | Example |
|---|---|---|
| `affirmed` | Default | *"purulent discharge noted"* |
| `negated` | no / not / without / denies / unremarkable | *"no signs of infection"* |
| `uncertain` | ? / possible / query / ?early | *"? early wound breakdown"* |
| `hypothetical` | if / should / in case of | *"if signs of infection develop"* |
| `historical` | previously / prior to / last admission | *"previously had wound infection"* |

Assertion status is the primary mechanism for handling the negation problem in clinical text — e.g. *"antibiotics not required"* produces `ANTIBIOTIC:negated`, a negative signal, rather than a false positive SSI indicator.

### 5.4 Orthopaedic Custom Rules

The base MedSpaCy model is extended with patterns covering:
- Prosthesis-specific terminology: *periprosthetic*, *implant infection*, *loose prosthesis*, *joint aspiration*, *synovial fluid*, *DAIR* (debridement, antibiotics, implant retention)
- NHS UK spelling variants: *haematoma*, *colour*, *anaesthetic*, *paracetamol*
- Common dictation artefacts and formatting inconsistencies in orthopaedic clinic letters
- Negative outcome phrases: *"wound is satisfactory"*, *"healing well"*, *"no concerns"*

Entity rules are defined in `src/ner/entity_rules.py` and are intended to be extended by clinical teams following MDT review feedback.

---

## 6. Classifier

### 6.1 Model Backbone

**Primary:** `Simonlee711/Clinical_ModernBERT` — a ModernBERT-base model pretrained on PubMed abstracts (~40M documents), MIMIC-IV clinical notes, and ICD-9 coded descriptions (13B tokens total). Selected for:
- **8,192 token context window** — eliminates sliding window complexity for long clinical documents; most NHS clinic letters and discharge summaries fit in a single forward pass
- **Flash Attention + RoPE** — faster CPU inference than standard BERT for sequences >256 tokens
- Clinical and biomedical pretraining corpus including MIMIC-IV (NHS-adjacent clinical language priors)
- ~100M parameters — comparable to BERT-base; runs on CPU for batch inference on NHS workstations
- MIT licence

**Comparison baseline:** `emilyalsentzer/Bio_ClinicalBERT` — trained on the same fine-tuning data for ablation comparison. Evaluating both models in the paper quantifies the architectural gain from the ModernBERT design and contributes downstream task benchmarks for Clinical_ModernBERT to the community (currently absent from the model card).

### 6.2 Input Construction

Structured metadata is prepended as special tokens before the note text:

```
[CLS] [PROCEDURE: hip_total] [DAYS_POST_OP: 12] [WINDOW: in_30d] <note text> [SEP]
```

This conditions the classification on known clinical context without a separate feature engineering step. The 8,192 token context window of Clinical_ModernBERT accommodates the full text of the vast majority of orthopaedic clinic letters and discharge summaries without truncation. Notes exceeding 8,192 tokens are truncated at sentence boundaries with a warning logged.

### 6.3 Output Head and Calibration

Linear layer over `[CLS]` token → 4 logits → softmax → calibrated probabilities:

```
P(none) + P(superficial) + P(deep) + P(organ_space) = 1.0
```

Temperature scaling is applied after fine-tuning to calibrate probabilities. Calibration is essential for the threshold-based MDT triage workflow — raw softmax scores are not reliable probability estimates without it.

### 6.4 ECDC Post-Softmax Gating

Deterministic rules applied after calibration enforce clinical validity:
- `days_post_op > 365`: zero and renormalise `P(deep)` and `P(organ_space)`
- `days_post_op > 30` AND no implant procedure (not applicable to current scope but included for future extensibility)
- These rules are unconditional — they cannot be overridden by model confidence

### 6.5 Threshold Logic

| Zone | Condition | Action |
|---|---|---|
| `auto_negative` | `P(none) ≥ 0.85` | No SSI — excluded from review list |
| `review_required` | `0.40 ≤ max(SSI classes) < 0.85` | Flagged for MDT review |
| `auto_positive` | `max(SSI classes) ≥ 0.85` | Classified — included in line list |

All thresholds are configurable in `config.yaml`. Surveillance teams can adjust the sensitivity/specificity trade-off for their local context without touching code.

### 6.6 Structured-Only Mode (ICD-10 Rule Engine)

When no free text is available, a deterministic rule engine classifies episodes by ICD-10 code presence:

| ICD-10 Code(s) | SSI Signal |
|---|---|
| `T81.4` | Infection following procedure |
| `T84.50`–`T84.54` | Prosthetic joint infection (deep / organ-space) |
| `T84.6` | Infection due to internal fixation device |
| `L02`, `L03` | Superficial wound infection / cellulitis |
| `M00.8`, `M00.9` | Pyogenic arthritis (organ-space proxy) |

Combined with ECDC window gating and OPCS-4 procedure validation, this produces a classification. Sensitivity is expected to be substantially lower than text-based classification, consistent with published literature (AUC ~0.55 for ICD codes vs ~0.99 for free text). This comparison is a primary output of the evaluation notebooks.

---

## 7. Input Data Specification

### 7.1 Mode 1 — Pre-concatenated Text

| Column | Type | Required | Description |
|---|---|---|---|
| `patient_id` | string | ✓ | Pseudonymised identifier (e.g. hashed NHS number) |
| `episode_id` | string | recommended | Unique episode identifier for linkage |
| `operation_date` | YYYY-MM-DD | ✓ | Date of index surgical procedure |
| `note_date` | YYYY-MM-DD | ✓ | Date this note was written |
| `procedure_code` | OPCS-4 | ✓ | See Section 7.3 |
| `note_text` | string | ✓ | Full free text of the clinical note |
| `icd10_codes` | pipe-separated | optional | e.g. `Z96.6\|M16.1` — for future hybrid mode |
| `hospital_site` | string | optional | Site identifier for multi-site stratification |

### 7.2 Mode 2 — Multiple Named Text Columns (administrative data style)

All columns from Mode 1 apply, with `note_text` replaced by any combination of:

| Column | Required | Description |
|---|---|---|
| `presenting_complaint` | optional | Chief complaint or reason for attendance |
| `clinical_findings` | optional | Examination findings, wound assessment |
| `diagnosis` | optional | Stated diagnosis or impression |
| `management_plan` | optional | Treatment plan, follow-up instructions |
| `discharge_summary` | optional | Discharge narrative |

At least one text column must be non-null per row. The pipeline concatenates columns in the order defined in `config.yaml`, inserting section headers between blocks. Rows with all text fields null are flagged `insufficient_data`.

Concatenation order is configurable:

```yaml
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
```

### 7.3 OPCS-4 Procedure Codes in Scope

| Code | Description |
|---|---|
| `W37` | Total prosthetic replacement of hip joint NEC |
| `W38` | Total prosthetic replacement of hip joint using cement |
| `W39` | Prosthetic replacement of head of femur NEC |
| `W40` | Prosthetic replacement of head of femur using cement |
| `W41` | Revision of prosthetic replacement of hip joint NEC |
| `W42` | Revision of prosthetic replacement of hip joint using cement |
| `W43` | Primary total prosthetic replacement of knee joint NEC |
| `W44` | Primary total prosthetic replacement of knee joint using cement |
| `W45` | Revision of prosthetic replacement of knee joint NEC |
| `W46` | Revision of prosthetic replacement of knee joint using cement |
| `W47` | Other prosthetic replacement of knee joint |

Episodes with procedure codes outside this list are flagged `out_of_scope` in the output rather than silently dropped.

### 7.4 Bad-Row Handling

| Condition | Flag Applied |
|---|---|
| `operation_date` missing | `missing_operation_date` — excluded from ECDC gating |
| All text fields null | `insufficient_data` — excluded from NER/classifier |
| Procedure code out of scope | `out_of_scope` — passed through to output with flag |
| `days_post_op` outside all ECDC windows | `outside_window` — included with flag, no SSI classification |

All flagged rows appear in the full line list with their flag in `ssi_classification`. None are silently discarded.

---

## 8. Output Format

### 8.1 Full Line List (`ssi_linelist_YYYYMMDD.csv`)

One row per input episode:

| Column | Description |
|---|---|
| `patient_id` | From input |
| `episode_id` | From input |
| `operation_date` | From input |
| `note_date` | From input |
| `days_post_op` | Computed from operation_date and note_date |
| `procedure_code` | From input |
| `procedure_description` | Looked up from OPCS-4 reference table |
| `hospital_site` | From input |
| `operating_mode` | `text_only` / `structured_only` / `hybrid` |
| `ssi_classification` | `none` / `superficial` / `deep` / `organ_space` / `out_of_scope` / `missing_operation_date` / `insufficient_data` / `outside_window` |
| `p_none` | Calibrated probability (0–1) |
| `p_superficial` | Calibrated probability (0–1) |
| `p_deep` | Calibrated probability (0–1) |
| `p_organ_space` | Calibrated probability (0–1) |
| `confidence_zone` | `auto_negative` / `review_required` / `auto_positive` |
| `review_required` | Boolean |
| `extracted_entities` | Pipe-separated entity:assertion pairs |
| `entity_snippets` | Pipe-separated text spans that triggered each entity |
| `ecdc_window_flag` | `within_30d` / `within_1yr` / `outside_window` |
| `icd10_codes` | From input, passed through |

### 8.2 MDT Review List (`ssi_review_YYYYMMDD.csv`)

Subset of the full line list where `review_required = True`. Sorted by maximum SSI class probability descending. Includes a blank `reviewer_notes` column for clinical team annotation. This file is the primary deliverable for MDT review workflow.

### 8.3 Run Summary (`ssi_summary_YYYYMMDD.txt`)

Plain text suitable for pasting into a surveillance report. Includes episode counts, classification breakdown with percentages and 95% confidence intervals, review-required count, operating mode, and threshold settings applied.

---

## 9. Training Data

### 9.1 MIMIC-IV-Note Silver Labels (Training)

**Source:** MIMIC-IV-Note v2.2 (PhysioNet). Requires free registration, CITI training completion, and data use agreement. Access instructions in `training/README.md`.

**Cohort construction:**
1. Filter MIMIC-IV by ICD-10 codes indicating postoperative infection: `T81.4`, `T84.5x`, `T84.6`
2. Filter by orthopaedic procedure codes (ICD-10-PCS or OPCS-4 equivalents)
3. Link to MIMIC-IV-Note discharge summaries and clinical notes for matched episodes
4. Apply ECDC sub-type heuristics to silver-label positive cases as superficial / deep / organ_space based on note section content and structured fields
5. Negative cases: episodes with orthopaedic procedures and no SSI-related codes, sampled to correct class imbalance

**Known limitation:** ICD-10 SSI coding in administrative data has published sensitivity ~10-50%. Silver labels will be noisy. The gold-standard evaluation set (Section 9.2) is the authoritative performance measure.

### 9.2 Gold-Standard Evaluation Set (Manual Annotation)

A held-out set of ~200–500 notes annotated by clinical colleagues using ECDC criteria as the annotation guide (provided in `data/annotations/annotation_guide.md`). Inter-annotator agreement is measured with Cohen's kappa prior to use as evaluation data.

This set is the primary basis for all performance metrics reported in the associated paper.

### 9.3 Pre-trained Models

| Model | HuggingFace ID | Role |
|---|---|---|
| Clinical_ModernBERT | `Simonlee711/Clinical_ModernBERT` | Primary classifier backbone |
| Bio+ClinicalBERT | `emilyalsentzer/Bio_ClinicalBERT` | Ablation comparison baseline |
| MedSpaCy | `en_core_sci_sm` (scispaCy) | NER base model |

---

## 10. Evaluation Framework

### 10.1 Primary Metrics (Surveillance-oriented)

- **Sensitivity (Recall):** critical for surveillance — missed cases undercount true SSI burden
- **Specificity:** controls false positive rate — affects MDT workload
- **Negative Predictive Value (NPV):** key for auto-negative zone — what proportion of auto-negatives are truly SSI-free?
- **AUC-ROC:** overall discriminative performance across all thresholds

### 10.2 Secondary Metrics

- **PPV (Precision):** relevant for auto-positive zone and MDT burden
- **F1 per ECDC class:** superficial / deep / organ_space individually
- **Calibration (Brier score, reliability diagram):** validates that probability scores reflect true event rates

### 10.3 Subgroup Analyses

- By ECDC SSI type (superficial / deep / organ_space)
- By procedure type (hip total / hip revision / knee total / knee revision)
- By `days_post_op` band (0–30 days / 31–180 days / 181–365 days)
- By operating mode (`text_only` vs `structured_only`) — primary stakeholder comparison

### 10.4 Key Paper Finding

Notebook `06_structured_vs_text.ipynb` produces the performance comparison between `structured_only` and `text_only` modes. This is the primary evidence for stakeholder engagement around free text data linkage from NHS surgical systems to administrative datasets.

---

## 11. Repository Structure

```
bert_SSI/
│
├── README.md                          # Primary documentation (GitHub-facing)
├── LICENSE                            # Apache 2.0
├── config.yaml                        # All configurable parameters
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package install
│
├── data/
│   ├── README.md                      # Data acquisition instructions
│   ├── raw/                           # Never committed — gitignored
│   ├── processed/                     # Intermediate features — gitignored
│   ├── annotations/
│   │   ├── annotation_guide.md        # ECDC criteria annotation instructions
│   │   └── schema.json                # Entity schema definition
│   └── reference/
│       ├── opcs4_orthopaedic.csv      # OPCS-4 code reference table
│       └── icd10_ssi_codes.csv        # ICD-10 SSI signal code reference
│
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── text_cleaner.py
│   │   ├── temporal.py
│   │   ├── concatenator.py
│   │   └── validator.py
│   ├── ner/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── entity_rules.py
│   │   └── assertion.py
│   ├── classifier/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── calibration.py
│   │   ├── ecdc_gating.py
│   │   └── structured.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── run.py
│   └── output/
│       ├── __init__.py
│       ├── formatter.py
│       └── summary.py
│
├── training/
│   ├── README.md
│   ├── mimic_silver_labels.py
│   ├── ecdc_heuristics.py
│   ├── train.py
│   └── evaluate.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_silver_label_analysis.ipynb
│   ├── 03_ner_development.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_structured_vs_text.ipynb
│
├── scripts/
│   ├── run_pipeline.py
│   └── annotate.py
│
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
│       └── fixtures/
│           └── synthetic_notes.csv
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
└── docs/
    └── design/
        └── 2026-03-19-bert-ssi-design.md
```

---

## 12. Testing and CI

### 12.1 Unit Tests

Each module is tested in isolation with mocked inputs. Synthetic note fixtures cover all assertion states:

- `affirmed_superficial`: *"Wound inspection shows erythema and purulent discharge at the incision site."*
- `negated_superficial`: *"No signs of infection. Wound healing well. Antibiotics not required."*
- `uncertain_deep`: *"Possible early wound breakdown at deep layer. Query dehiscence."*
- `affirmed_organ_space`: *"Joint aspiration confirmed periprosthetic infection with culture positive for Staph aureus."*
- `negated_all`: *"Satisfactory post-operative recovery. No concerns regarding the surgical site."*
- `historical`: *"Patient previously had a wound infection following the contralateral procedure."*

### 12.2 Smoke Tests

Full pipeline end-to-end runs on a synthetic 20-row CSV, one per operating mode. Validate pipeline completion, output file production, column presence, label validity, and probability sum ≈ 1.0. Do not validate classification accuracy.

### 12.3 GitHub Actions CI

Triggers on push and pull request to `main`. Matrix: Python 3.9, 3.10, 3.11. Steps: lint (flake8), format check (black), unit tests, smoke tests, coverage upload (Codecov).

---

## 13. Licence and Attribution

**Licence:** Apache 2.0 — permissive, compatible with NHS and public sector reuse, allows commercial adaptation.

**Training data:** MIMIC-IV-Note (PhysioNet Credentialed Health Data Licence). Users must obtain their own PhysioNet access. No MIMIC data is committed to this repository.

**Pre-trained models:** Clinical_ModernBERT (MIT licence); Bio+ClinicalBERT (MIT licence). Both available via HuggingFace.

---

## 14. Key References

- van Mourik MSM et al. PRAISE: providing a roadmap for automated infection surveillance in Europe. *Clin Microbiol Infect.* 2021;27(S1):S3–S19. doi:10.1016/j.cmi.2021.02.028
- PRAISE SSI Working Group. Automated surveillance for surgical site infections — expert perspectives for implementation. *Antimicrob Resist Infect Control.* 2024. PMC11667888.
- Bucher et al. Portable Automated Surveillance of Surgical Site Infections Using NLP. *Ann Surg.* 2020. PMC9040555.
- Danish DL study. Assessing the utility of deep neural networks in detecting superficial SSI from free text EHR data. PMC10801170.
- Alsentzer et al. Publicly Available Clinical BERT Embeddings. NAACL 2019. (Bio+ClinicalBERT)
- Warner et al. ModernBERT. 2024. (answerdotai/ModernBERT-base)
- Lee S. Clinical_ModernBERT. HuggingFace. 2025. (Simonlee711/Clinical_ModernBERT)
- ECDC HAI-Net SSI Protocol v2.2. European Centre for Disease Prevention and Control.
- UKHSA Protocol for the Surveillance of Surgical Site Infection. UK Health Security Agency.
