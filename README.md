# llm-audit-transfer-study

---

# Lightweight Output-Only Auditors Under Distribution Shift

**An empirical study of transfer limits for scalable LLM safety auditing**

---

## Overview

This project investigates whether **lightweight, scalable, output-only auditors** can generalize under distribution shift to detect unsafe language produced by a language model. Specifically, we study whether simple lexical auditors trained on some families of harmful outputs can transfer to **unseen attack families**, including both semantic shifts (new types of harmful intent) and obfuscation-style shifts (encoded or transformed text).

The work is framed as a small, CPU-only empirical study intended to probe **alignment-relevant failure modes** under realistic constraints: no access to prompts, no model internals, and no expensive representations.

---

## Motivation

Scalable AI safety interventions often rely on **output-based monitoring** rather than full access to model internals or prompts. In deployment settings, safety systems may need to:

* Audit model behavior using only generated text
* Generalize to **novel or evolving attack styles**
* Operate cheaply and transparently (e.g., linear models, simple features)

This project asks whether such lightweight auditors can meaningfully generalize under distribution shift, and where they fail.

Key motivations:

* Output-only auditing is often the *only* feasible safety signal at scale
* Transfer to unseen attacks is critical for robustness
* Negative results are highly informative for alignment research when diagnosed carefully

---

## Research Questions

This project addresses three related questions:

1. **Semantic transfer:**
   Can an output-only auditor trained on one harmful family (e.g., authority-based coercion) generalize to a *different* harmful family (e.g., direct harm requests) in plain English?

2. **Sample efficiency:**
   If zero-shot transfer fails, does limited in-family exposure (e.g., 20% of a new family) substantially improve performance?

3. **Obfuscation robustness:**
   Do lightweight lexical representations fail under encoding-style distribution shift, even when character-level structure is preserved?

---

## Experimental Setup

### Base Model

* **google/flan-t5-base**
* CPU-only inference
* Instruction-tuned but not strongly refusal-aligned in this setup

### Prompt Families

Each prompt belongs to one of the following families:

* **benign** (label = 0)
* **direct_harm** (label = 1)
* **authority** (label = 1)
* **encoding** (label = 1, with benign encoded controls)

Prompts are run through the base model to generate outputs.
**Auditors see only `output_text` — never prompts.**

---

## Output-Based Proxy Labeling

Each generated output is assigned a proxy label `y`:

* `y = 0` if output is benign or contains an explicit refusal
* `y = 1` if output complies with a harmful prompt

A conservative phrase-based heuristic is used to detect refusals.

---

## Important Pivot: Absence of Refusal Behavior

The project originally aimed to study **compliance vs refusal discrimination**. However:

* The base model produced **zero refusals**, even after:

  * adding explicit refusal-eliciting prompts
  * beam search decoding
  * stochastic sampling with temperature/top-p
* As a result, the dataset contained no `(prompt_label=1, y=0)` examples.

**Conclusion:**
The original framing was empirically unsupported for this model.

**Pivot:**
The project was reframed to study **transfer of output-only harm-topic auditors under distribution shift**, preserving alignment relevance without changing the base model.

This pivot — and the diagnostics leading to it — are a core contribution of the work.

---

## Auditor Models

All auditors use the same simple architecture:

* **Feature extraction:** TF-IDF

  * Word n-grams (1–2)
  * Character n-grams (3–5) for ablation
* **Classifier:** Logistic Regression
* **No prompt access, no model internals**

This design intentionally prioritizes:

* interpretability
* scalability
* minimal assumptions

---

## Experiments and Results

### 1. Obfuscation Transfer (Encoding Holdout)

**Setup:**

* Train on benign + direct_harm + authority
* Test on encoding family (with benign encoded controls)

**Results:**

* Very low recall on unsafe outputs
* Auditor predicts “safe” for most encoded harmful text
* Character n-grams do **not** recover performance

**Conclusion:**
Lightweight lexical auditors fail catastrophically under obfuscation-style distribution shift, even when character statistics are preserved.

---

### 2. Semantic Transfer: Holdout Direct Harm (0% Exposure)

**Setup:**

* Train on benign + authority
* Test on direct_harm + benign controls

**Results:**

* Test accuracy ≈ **0.79**
* Harmful recall ≈ **0.80**
* Few false negatives and false positives

**Conclusion:**
Auditors generalize well to an unseen **semantic attack family** when language is plain English.

---

### 3. Semantic Transfer: Holdout Authority (0% Exposure)

**Setup:**

* Train on benign + direct_harm
* Test on authority + benign controls

**Results:**

* Test accuracy ≈ **0.64**
* Harmful recall ≈ **0.60**
* Higher false-negative rate than direct_harm holdout

**Conclusion:**
Semantic transfer is **asymmetric**. Some harmful families rely on distinct lexical cues that do not fully transfer zero-shot.

---

### 4. Limited Exposure: Authority (20% Training Exposure)

**Setup:**

* Same as above, but allow 20% of authority examples into training

**Results:**

* Test accuracy improves to ≈ **0.75**
* Harmful recall improves to ≈ **0.75**
* False negatives drop substantially

**Conclusion:**
Auditors are **sample-efficient adapters**: small in-family exposure significantly improves performance.

---

## Key Takeaways

* Output-only lexical auditors **can generalize across unseen semantic attack families**
* Transfer is **not symmetric** and depends on linguistic overlap
* **Obfuscation breaks transfer entirely**, even with character-level features
* Limited in-family exposure enables rapid recovery
* Lightweight auditors learn **surface regularities**, not deep semantic invariants

---

## Limitations

* No refusal behavior observed in base model
* Lexical representations only (no embeddings or neural encoders)
* Small dataset by design
* No adversarial decoding or transformation-aware modeling

These limitations are intentional and define the scope of the study.

---

## Reproducibility

### Pipeline

1. Generate outputs

   ```bash
   python src/run_inference.py
   ```
2. Build auditor dataset

   ```bash
   python src/build_auditor_dataset.py
   ```
3. Train and evaluate auditors

   ```bash
   python src/train_auditor.py
   ```

### Key Config Options (in `train_auditor.py`)

* `features = "word"` or `"char"`
* `holdout_family = "direct_harm" | "authority" | "encoding"`
* `holdout_train_fraction = 0.0` (true transfer) or `> 0.0` (limited exposure)

---

## Why This Matters for AI Safety

This project demonstrates that **scalable, output-only auditing is feasible but fragile**. While simple auditors can generalize across nearby semantic threats, they fail under obfuscation — a realistic adversarial strategy. Robust safety monitoring likely requires either stronger representations, explicit deobfuscation mechanisms, or complementary defenses.

Equally important, this work highlights the value of **honest negative results** and principled pivots when initial alignment hypotheses are not supported by empirical evidence.

---

## Author

Luke Coffman
AI Safety / Machine Learning
(Anthropic AI Safety Fellowship Application Artifact)

---
