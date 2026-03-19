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

Since all in-scope procedures (hip/knee arthroplasty, W37–W47) involve prosthetic implants, deep incisional and organ/space SSI surveillance windows extend to **1 year post-operatively**. `days_post_op` is computed from `operation_date` and `note_date` and gates which ECDC classes are clinically possible prior to model scoring. See Section 7.5 for `days_post_op` computation and edge case handling.

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
│  • Multi-column text concatenation (input format B)             │
│  • days_post_op computation with edge case handling             │
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

### 4.2 Operating Modes and Input Format Variants

Two orthogonal choices determine pipeline behaviour:

**Processing mode** (how the episode is classified):

| Processing Mode | Condition | Components Active |
|---|---|---|
| `text_only` | ≥1 text column present | Preprocessor → NER + classifier → Output |
| `structured_only` | No text columns present; `icd10_codes` present | Preprocessor → ICD-10 rule engine → Output |
| `hybrid` *(future)* | Both text and `icd10_codes` present | All components |

**Input format** (how text columns are structured):

| Input Format | Columns | Notes |
|---|---|---|
| Format A | Single `note_text` column | Pre-concatenated; used directly |
| Format B | Multiple named text columns | Concatenated by pipeline in configured order |

Both Format A and Format B can feed into `text_only` or `hybrid` processing modes. Processing mode is auto-detected from column presence, or set explicitly in `config.yaml` via `processing_mode`.

The `structured_only` mode deliberately replicates the capability of existing administrative data-based surveillance, providing a direct performance comparator against `text_only` mode.

---

## 5. NER Layer

### 5.1 Framework

`MedSpaCy` (Python package, installed via `pip install medspacy`) with the `ConText` algorithm for assertion detection. MedSpaCy extends spaCy and does not have a HuggingFace model card — it is installed as a Python package, not downloaded via the HuggingFace Hub.

The underlying spaCy language model used for tokenisation and base NER is `en_core_sci_sm` from the `scispaCy` package (installed separately via `pip install scispacy` and the scispaCy release URL). MedSpaCy wraps this spaCy pipeline and adds the ConText component and custom rule matchers on top. The relationship is: `scispaCy (en_core_sci_sm)` → base tokenisation and entity recognition → `MedSpaCy ConText` → assertion status → custom `entity_rules.py` patterns → final entity list.

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

**Expected hardware and throughput:** On a standard NHS workstation (Intel Core i7, 16 GB RAM, CPU only), estimated throughput is approximately 8–15 notes per minute for Clinical_ModernBERT with the 8,192 token window. A weekly batch of 500 orthopaedic episodes would complete in approximately 35–60 minutes. Minimum RAM: 8 GB (model + tokeniser + batch overhead). For users with access to a GPU (CUDA), throughput increases to approximately 100–200 notes per minute. These figures should be updated with benchmarks on reference hardware prior to publication.

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

**Temperature scaling** is applied after fine-tuning. A dedicated calibration split (10% of training data, held out from fine-tuning) is used for temperature optimisation. The calibration split must not overlap with the gold-standard evaluation set. Temperature scaling is implemented in `src/classifier/calibration.py`. Calibration is essential for the threshold-based MDT triage workflow — raw softmax scores are not reliable probability estimates without it.

### 6.4 ECDC Post-Softmax Gating

Deterministic rules applied after calibration enforce clinical validity. For all in-scope procedures (W37–W47), implant presence is derived from `procedure_code` — all codes in scope involve prosthetic implants, so `implant = True` for the entire current scope.

Gating rules:
- `days_post_op > 365`: zero `P(deep)` and `P(organ_space)`, renormalise remaining probabilities
- `days_post_op > 30` AND `implant = False`: zero all SSI classes, set `P(none) = 1.0` (not applicable to current W37–W47 scope; included for extensibility when non-implant procedure codes are added)

These rules are unconditional — they cannot be overridden by model confidence.

### 6.5 Threshold Logic

The three zones are evaluated in priority order. `review_required` is the catch-all for any row that clears neither auto-threshold:

| Priority | Zone | Condition | `confidence_zone` | `review_required` |
|---|---|---|---|---|
| 1 | `auto_negative` | `P(none) ≥ 0.85` | `auto_negative` | `False` |
| 2 | `auto_positive` | `max(P(superficial), P(deep), P(organ_space)) ≥ 0.85` | `auto_positive` | `False` |
| 3 (catch-all) | `review_required` | All other rows | `review_required` | `True` |

`review_required` is a Boolean column derived from `confidence_zone == 'review_required'`. The two columns are always consistent by construction. The catch-all ensures every row is assigned a zone regardless of probability distribution shape — including low-confidence rows where `P(none) < 0.85` and `max(SSI) < 0.40`.

All thresholds are configurable in `config.yaml`. Surveillance teams can adjust the sensitivity/specificity trade-off for their local context without touching code.

### 6.6 Structured-Only Mode (ICD-10 Rule Engine)

When no free text is available, a deterministic rule engine classifies episodes using **NHS ICD-10 (WHO fifth edition)** codes as recorded in HES, PEDW, and SMR. Note: these differ from ICD-10-CM (US clinical modification), which uses additional fifth-character subdivisions. The MIMIC-IV training data uses ICD-10-CM; this distinction is handled explicitly in the training pipeline (see Section 9.1).

| NHS ICD-10 Code | SSI Signal |
|---|---|
| `T81.4` | Infection following procedure |
| `T84.5` | Infection/inflammatory reaction due to internal joint prosthesis |
| `T84.6` | Infection due to internal fixation device |
| `L02` | Cutaneous abscess / furuncle (superficial) |
| `L03` | Cellulitis and acute lymphangitis (superficial) |
| `M00.8` | Arthritis due to specified bacterial agents (organ-space proxy) |
| `M00.9` | Pyogenic arthritis, unspecified (organ-space proxy) |

Combined with ECDC window gating and OPCS-4 procedure validation, this produces a classification. Sensitivity is expected to be substantially lower than text-based classification, consistent with published literature (AUC ~0.55 for ICD codes vs ~0.99 for free text). This comparison is a primary output of the evaluation notebooks.

---

## 7. Input Data Specification

### 7.1 Format A — Pre-concatenated Text

| Column | Type | Required | Description |
|---|---|---|---|
| `patient_id` | string | ✓ | Pseudonymised identifier (e.g. hashed NHS number) |
| `episode_id` | string | ✓ | Unique episode identifier for clinical linkage — see Section 7.4 |
| `operation_date` | YYYY-MM-DD | ✓ | Date of index surgical procedure |
| `note_date` | YYYY-MM-DD | ✓ | Date this note was written |
| `procedure_code` | OPCS-4 | ✓ | See Section 7.3 |
| `note_text` | string | ✓ | Full free text of the clinical note |
| `icd10_codes` | pipe-separated | optional | e.g. `Z96.6\|M16.1` — for future hybrid mode |
| `hospital_site` | string | optional | Site identifier for multi-site stratification |

### 7.2 Format B — Multiple Named Text Columns (administrative data style)

All columns from Format A apply, with `note_text` replaced by any combination of:

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

All W37–W47 codes involve prosthetic implants; the 1-year ECDC surveillance window applies to all.

| Code | Description | Procedure type |
|---|---|---|
| `W37` | Total prosthetic replacement of hip joint NEC | Hip total |
| `W38` | Total prosthetic replacement of hip joint using cement | Hip total |
| `W39` | Prosthetic replacement of head of femur NEC | Hip hemi |
| `W40` | Prosthetic replacement of head of femur using cement | Hip hemi |
| `W41` | Revision of prosthetic replacement of hip joint NEC | Hip revision |
| `W42` | Revision of prosthetic replacement of hip joint using cement | Hip revision |
| `W43` | Primary total prosthetic replacement of knee joint NEC | Knee total |
| `W44` | Primary total prosthetic replacement of knee joint using cement | Knee total |
| `W45` | Revision of prosthetic replacement of knee joint NEC | Knee revision |
| `W46` | Revision of prosthetic replacement of knee joint using cement | Knee revision |
| `W47` | Other prosthetic replacement of knee joint | Knee other |

W39 and W40 (hemiarthroplasty — prosthetic replacement of the femoral head) are included in scope. These procedures are mechanistically distinct from total hip replacement (commonly performed for fractured neck of femur). Performance is reported as a separate evaluation subgroup (Section 10.3).

Episodes with procedure codes outside this list are flagged `out_of_scope` in the output rather than silently dropped.

### 7.4 `episode_id` Requirement and Fallback

`episode_id` is a **required** column. It is used by MDT reviewers to retrieve the original episode from clinical systems; a null `episode_id` in the review list renders it clinically unusable. If `episode_id` is absent from the input, the pipeline will raise a validation error and halt before processing. A surrogate of `patient_id` + `note_date` is not an acceptable substitute because it does not guarantee uniqueness across multi-episode patients.

### 7.5 `days_post_op` Computation and Edge Cases

`days_post_op` is computed as `(note_date - operation_date).days` (integer, calendar days).

| Condition | Behaviour |
|---|---|
| `note_date == operation_date` | `days_post_op = 0` — valid; same-day notes (e.g. intraoperative) are within the ECDC window |
| `note_date < operation_date` | Row flagged `invalid_dates`, excluded from classification; warning logged |
| `operation_date` null | Row flagged `missing_operation_date`, excluded from classification |
| `note_date` null | Row flagged `missing_note_date`, excluded from classification |

### 7.6 Bad-Row Handling Summary

| Condition | Flag Applied |
|---|---|
| `operation_date` missing | `missing_operation_date` |
| `note_date` missing | `missing_note_date` |
| `note_date < operation_date` | `invalid_dates` |
| All text fields null (text modes) | `insufficient_data` |
| Procedure code out of scope | `out_of_scope` |
| `days_post_op > 365` | `outside_window` |

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
| `procedure_type` | Derived: `hip_total` / `hip_hemi` / `hip_revision` / `knee_total` / `knee_revision` / `knee_other` |
| `hospital_site` | From input |
| `processing_mode` | `text_only` / `structured_only` / `hybrid` |
| `ssi_classification` | `none` / `superficial` / `deep` / `organ_space` / `out_of_scope` / `missing_operation_date` / `missing_note_date` / `invalid_dates` / `insufficient_data` / `outside_window` |
| `p_none` | Calibrated probability (0–1); null for `structured_only` mode |
| `p_superficial` | Calibrated probability (0–1); null for `structured_only` mode |
| `p_deep` | Calibrated probability (0–1); null for `structured_only` mode |
| `p_organ_space` | Calibrated probability (0–1); null for `structured_only` mode |
| `confidence_zone` | `auto_negative` / `review_required` / `auto_positive`; `rule_based` for `structured_only` |
| `review_required` | Boolean — derived from `confidence_zone == 'review_required'` |
| `extracted_entities` | Pipe-separated entity:assertion pairs; null for `structured_only` mode |
| `entity_snippets` | Pipe-separated text spans that triggered each entity; null for `structured_only` mode |
| `ecdc_window_flag` | `within_30d` / `within_1yr` / `outside_window` |
| `icd10_codes` | From input, passed through |

### 8.2 MDT Review List (`ssi_review_YYYYMMDD.csv`)

Subset of the full line list where `review_required = True`. Sorted by maximum SSI class probability descending. Includes a blank `reviewer_notes` column for clinical team annotation. This file is the primary deliverable for MDT review workflow.

### 8.3 Run Summary (`ssi_summary_YYYYMMDD.txt`)

Plain text suitable for pasting into a surveillance report. Includes episode counts, classification breakdown with percentages and 95% confidence intervals, review-required count, processing mode, and threshold settings applied.

---

## 9. Training Data

### 9.1 MIMIC-IV-Note Silver Labels (Training)

**Source:** MIMIC-IV-Note v2.2 (PhysioNet). Requires free registration, CITI training completion, and data use agreement. Access instructions in `training/README.md`.

**Note on coding systems:** MIMIC-IV uses ICD-10-CM (US clinical modification), which includes fifth-character subdivisions not present in NHS ICD-10 (WHO). The training cohort filter uses ICD-10-CM codes (`T84.50`–`T84.54` for prosthetic joint infection); the `structured_only` inference mode uses NHS ICD-10 (`T84.5`). These are clinically equivalent but syntactically different. The training pipeline (`training/mimic_silver_labels.py`) documents this distinction explicitly and uses ICD-10-CM throughout. The inference rule engine (`src/classifier/structured.py`) uses NHS ICD-10 throughout.

**Cohort construction:**
1. Filter MIMIC-IV by ICD-10-CM codes indicating postoperative infection: `T81.4`, `T84.50`–`T84.54`, `T84.6`
2. Filter by orthopaedic procedure codes (ICD-10-PCS equivalents of OPCS-4 W37–W47)
3. Link to MIMIC-IV-Note discharge summaries and clinical notes for matched episodes
4. Apply ECDC sub-type heuristics to silver-label positive cases as superficial / deep / organ_space based on note section content and structured fields
5. Negative cases: episodes with orthopaedic procedures and no SSI-related codes, sampled to correct class imbalance
6. Reserve 10% as calibration split (for temperature scaling); remainder used for fine-tuning

**Known limitation:** ICD-10 SSI coding in administrative data has published sensitivity ~10-50%. Silver labels will be noisy. The gold-standard evaluation set (Section 9.2) is the authoritative performance measure.

### 9.2 Gold-Standard Evaluation Set (Manual Annotation)

A held-out set of ~200–500 notes annotated by clinical colleagues using ECDC criteria as the annotation guide (provided in `data/annotations/annotation_guide.md`). The evaluation set is entirely separate from the MIMIC-IV training and calibration data.

**Annotation protocol:**
- Minimum 2 independent annotators per note
- Inter-annotator agreement measured with Cohen's kappa; minimum acceptable kappa: **0.70** (substantial agreement) before use as evaluation data
- Disagreements below this threshold are adjudicated by a senior clinician applying ECDC criteria
- Adjudication decisions are documented in `data/annotations/adjudication_log.csv`

This set is the primary basis for all performance metrics reported in the associated paper.

### 9.3 Pre-trained Models and Installation

| Model | Identifier | Installation | Role |
|---|---|---|---|
| Clinical_ModernBERT | `Simonlee711/Clinical_ModernBERT` | `pip install transformers` then `AutoModel.from_pretrained(...)` via HuggingFace Hub | Primary classifier backbone |
| Bio_ClinicalBERT | `emilyalsentzer/Bio_ClinicalBERT` | As above | Ablation comparison baseline |
| MedSpaCy | n/a (Python package) | `pip install medspacy` | ConText assertion detection framework |
| scispaCy `en_core_sci_sm` | n/a (spaCy model) | `pip install scispacy` + release URL | Base tokenisation and NER for MedSpaCy |

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
- By procedure type (hip total / hip hemi / hip revision / knee total / knee revision / knee other)
- By `days_post_op` band (0–30 days / 31–180 days / 181–365 days)
- By processing mode (`text_only` vs `structured_only`) — primary stakeholder comparison

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
│   │   ├── adjudication_log.csv       # Annotator disagreement records
│   │   └── schema.json                # Entity schema definition
│   └── reference/
│       ├── opcs4_orthopaedic.csv      # OPCS-4 code reference table
│       └── icd10_ssi_codes.csv        # NHS ICD-10 SSI signal code reference
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

Full pipeline end-to-end runs on a synthetic 20-row CSV, one per processing mode. Validate pipeline completion, output file production, column presence, label validity, and probability sum ≈ 1.0. Do not validate classification accuracy.

### 12.3 GitHub Actions CI

Triggers on push and pull request to `main`. Matrix: Python 3.9, 3.10, 3.11. Steps: lint (flake8), format check (black), unit tests, smoke tests, coverage upload (Codecov).

---

## 13. Licence and Attribution

**Licence:** Apache 2.0 — permissive, compatible with NHS and public sector reuse, allows commercial adaptation.

**Training data:** MIMIC-IV-Note (PhysioNet Credentialed Health Data Licence). Users must obtain their own PhysioNet access. No MIMIC data is committed to this repository.

**Pre-trained models:** Clinical_ModernBERT (MIT licence); Bio_ClinicalBERT (MIT licence). Both available via HuggingFace.

---

## 14. Key References

- van Mourik MSM et al. PRAISE: providing a roadmap for automated infection surveillance in Europe. *Clin Microbiol Infect.* 2021;27(S1):S3–S19. doi:10.1016/j.cmi.2021.02.028
- PRAISE SSI Working Group. Automated surveillance for surgical site infections — expert perspectives for implementation. *Antimicrob Resist Infect Control.* 2024. PMC11667888.
- Bucher et al. Portable Automated Surveillance of Surgical Site Infections Using NLP. *Ann Surg.* 2020. PMC9040555.
- Danish DL study. Assessing the utility of deep neural networks in detecting superficial SSI from free text EHR data. PMC10801170.
- Alsentzer et al. Publicly Available Clinical BERT Embeddings. NAACL 2019. (Bio_ClinicalBERT)
- Warner et al. ModernBERT. 2024. (answerdotai/ModernBERT-base)
- Lee S. Clinical_ModernBERT. HuggingFace. 2025. (Simonlee711/Clinical_ModernBERT)
- ECDC HAI-Net SSI Protocol v2.2. European Centre for Disease Prevention and Control.
- UKHSA Protocol for the Surveillance of Surgical Site Infection. UK Health Security Agency.
