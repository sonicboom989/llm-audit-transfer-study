---
# REPORT: Transfer Limits of Lightweight Output-Only Auditors

## 1. Initial Objective and Framing

The original goal of this project was to study whether a **lightweight, output-only auditor** could distinguish **unsafe compliance from safe refusal** in language model outputs. The motivation was alignment-driven: scalable safety systems often lack access to prompts or model internals and must instead reason over generated text alone.

The intended setup was:

* Base model: `google/flan-t5-base`
* Auditor input: model outputs only (`output_text`)
* Target distinction:

  * harmful compliance (`y=1`)
  * safe refusal (`y=0`) for harmful prompts

This framing would have allowed study of whether an output-only classifier could learn refusal behavior as a safety signal.

---

## 2. Empirical Failure of the Original Framing

### 2.1 Absence of Refusal Behavior

After generating outputs for benign, direct_harm, and authority prompt families, I observed that the base model **never refused** any harmful prompts in the dataset.

To rule out superficial causes, I attempted multiple interventions:

* **Prompt augmentation:**
  Added explicitly malicious but non-operational prompts intended to elicit refusal.
* **Decoding changes:**

  * Beam search (`num_beams=4`, deterministic)
  * Stochastic sampling (temperature / top-p, shorter generations)

Despite these efforts, **zero outputs were classified as refusals** by a conservative phrase-based heuristic.

### 2.2 Implication for the Dataset

This resulted in a structural issue:

* All harmful prompts yielded `y=1` (unsafe compliance).
* There were no `(prompt_label=1, y=0)` counterexamples.

As a result, the auditor could not possibly learn a compliance-vs-refusal distinction. Any apparent performance would instead reflect **topic-level harmfulness**, not safety behavior.

This was not a modeling bug but a **data reality** given the base model’s behavior.

---

## 3. Decision to Pivot (and Why It Was Necessary)

At this point, continuing with the original framing would have produced misleading conclusions. Rather than change the base model or artificially inject refusals, I chose to **pivot the research question** while keeping the same infrastructure and constraints.

### New framing:

> **Can lightweight, output-only auditors trained on some harmful output families generalize to *unseen* attack families under distribution shift?**

This pivot preserves alignment relevance:

* Real-world auditors often encounter **novel or evolving attack styles**
* Robustness to distribution shift is a core safety concern
* The output-only constraint remains intact

Crucially, the pivot was **motivated by empirical evidence**, not convenience.

---

## 4. Revised Experimental Design

### 4.1 Prompt Families

Each prompt belongs to one family:

* `benign`
* `direct_harm`
* `authority`
* `encoding` (obfuscated text; includes benign encoded controls)

Prompts are run through the base model to generate outputs.
Auditors see **only output text**, never prompts.

### 4.2 Proxy Labeling

Each output is assigned a proxy label:

* `y = 0` if benign or refusal
* `y = 1` if harmful compliance

Given the absence of refusals, this reduces to a **harm-topic classifier**, which is explicitly acknowledged.

---

## 5. Obfuscation Failure: Encoding Holdout

### 5.1 Hypothesis

The encoding family (reversed text, Caesar shifts, base64-like strings) preserves character statistics but destroys word-level semantics. I hypothesized that:

* Word-level TF-IDF would fail
* Character n-grams might recover signal

### 5.2 Diagnostics

To test whether encoding outputs were simply “noise,” I computed:

* Average output length
* Alphabetic character ratio

These statistics were comparable between training and encoding test sets, indicating **structured text**, not randomness.

### 5.3 Results

* Word-level TF-IDF: near-total failure (high false negatives)
* Character n-grams (3–5): no improvement; performance worsened slightly

### 5.4 Conclusion

The failure is **not merely representational granularity**. Even subword lexical features fail under obfuscation-style distribution shift.

This suggests that lightweight lexical auditors lack invariance to simple transformations that preserve intent but alter surface form.

---

## 6. Semantic Family Transfer Experiments

To determine whether the encoding failure reflected a general inability to transfer, I conducted **leave-one-family-out** evaluations on plain-English families.

### 6.1 Holdout: Direct Harm (0% Exposure)

* Train: benign + authority
* Test: direct_harm + benign controls

**Result:**
Strong transfer (≈0.79 accuracy, ≈0.80 recall on harmful outputs).

**Interpretation:**
The auditor generalizes well to an unseen semantic family when language remains unencoded.

---

### 6.2 Holdout: Authority (0% Exposure)

* Train: benign + direct_harm
* Test: authority + benign controls

**Result:**
Partial transfer (≈0.64 accuracy, ≈0.60 recall).

**Interpretation:**
Transfer is asymmetric. Authority-based harmful language relies on lexical cues not fully captured from direct_harm training alone.

---

### 6.3 Limited Exposure: Authority (20%)

* Same setup, but allow 20% of authority examples into training

**Result:**
Performance improves substantially (≈0.75 accuracy, ≈0.75 recall).

**Interpretation:**
The auditor exhibits **sample-efficient adaptation**, suggesting reliance on family-specific surface regularities rather than deep semantic invariants.

---

## 7. Synthesis of Findings

Across all experiments, three regimes emerge:

1. **Semantic family shift (plain English):**
   Often transferable, though asymmetrically.
2. **Limited in-family exposure:**
   Rapidly improves performance.
3. **Obfuscation / encoding shift:**
   Causes catastrophic failure, even with character-level features.

This triangulation is more informative than any single result.

---

## 8. Limitations and What Would Be Needed Next

This study intentionally avoids:

* Neural embeddings
* Prompt access
* Model internals
* Larger datasets

As a result:

* The auditor learns **surface lexical regularities**, not robust intent.
* True compliance-vs-refusal learning would require:

  * a base model that actually refuses,
  * or explicit intervention to induce refusals,
  * or richer representations with semantic invariance.

These limitations define the scope rather than undermine the conclusions.

---

## 9. Alignment Relevance

This project highlights a key safety lesson:

> **Scalable, output-only auditors can work for nearby threat models but are brittle under obfuscation and require family coverage to maintain recall.**

Equally important, it demonstrates the importance of:

* diagnosing dataset pathologies,
* abandoning unsupported hypotheses,
* and pivoting toward empirically grounded questions.

---

## 10. Closing Note

This work is intentionally modest in scale but careful in reasoning. The primary contribution is not performance, but **clarity about what lightweight auditing can and cannot do under realistic constraints**.

The pivot itself is part of the result.

---

