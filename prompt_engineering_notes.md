# Prompt Engineering Notes — VLM Triage

## Context and objective

This document records the prompt-engineering decisions behind the frame-level triage step in `vlm_triage.py`. The goal is to map detector outputs into operationally useful severity labels (`LOW`, `MEDIUM`, `HIGH`) with concise rationale suitable for downstream review.

Model in production path: `claude-sonnet-4-20250514` via Anthropic Messages API.

---

## 1) Problem statement: why raw detections need model-based summarization

Raw detector rows in `detections.csv` are necessary but not sufficient for triage decisions:

- Detection logs are **box-level** and fragmented; reviewers need a frame-level safety interpretation.
- Confidence and class counts are present, but there is no explicit **risk semantics** (e.g., vulnerable road users mixed with vehicles).
- Rule filters can find candidate frames, but they over-trigger and do not prioritize urgency.
- Interview/demo requirement: produce an auditable severity class with short human-readable reasoning.

Design implication: convert each candidate frame into a compact textual scene summary, then ask the model for a constrained severity decision.

---

## 2) Iteration 1: naive prompt and observed weaknesses

### Prompt (v1)

```text
Given these detections, tell me how risky this frame is and explain.
<event description>
```

### What went wrong

- **Output shape instability**: responses varied between paragraphs, bullets, and mixed labels; parsing was brittle.
- **Label drift**: model produced synonyms (`critical`, `moderate`, `low risk`) instead of the required 3-class taxonomy.
- **Over-explanation**: long narrative responses increased token cost and reduced consistency.
- **Ambiguous grounding**: rationale sometimes referenced concepts not present in the summary (hallucinated context).

### Engineering takeaway

Natural-language-only instructions were too permissive for a pipeline that writes structured CSV outputs.

---

## 3) Iteration 2: structured-output prompt with explicit contract

### Prompt (v2)

```text
You are an AV safety triage analyst.
Classify severity as one of LOW, MEDIUM, HIGH.
Return JSON with keys:
- severity
- reasoning
Event:
<event description>
```

### Improvements

- Response parsing became mostly deterministic.
- Label space narrowed to the intended triage classes.
- CSV writing path became straightforward (`severity`, `reasoning` fields).

### Remaining issues

- Occasional non-JSON wrappers (preface text, markdown fences).
- Inconsistent rationale length (single phrase vs. long paragraph).
- Rare invalid severity values still required fallback handling.

### Engineering takeaway

A schema hint helps, but strictness must be increased further (format-only response + exact allowed values + brief reasoning length).

---

## 4) Final prompt (used in `vlm_triage.py`) and design rationale

### Final prompt text

```text
You are an AV safety triage analyst.
Given this frame-level event description, classify safety severity as exactly one of:
LOW, MEDIUM, HIGH.

Return valid JSON only with keys:
- "severity": one of LOW|MEDIUM|HIGH
- "reasoning": short explanation (1-3 sentences)

Event description:
<event_description>
```

### Why each decision exists

- **Role framing (`AV safety triage analyst`)**  
  Anchors responses toward safety operations rather than generic object-detection commentary.

- **Exact 3-class taxonomy (`exactly one of LOW, MEDIUM, HIGH`)**  
  Prevents severity vocabulary drift and supports stable aggregation in `triage_results.csv`.

- **`valid JSON only` contract**  
  Optimizes for machine-readability and direct `json.loads(...)` parsing.

- **Explicit key contract (`severity`, `reasoning`)**  
  Keeps output schema aligned with the result table columns and avoids optional/unexpected fields.

- **Reasoning length bound (`1-3 sentences`)**  
  Produces concise justifications appropriate for triage dashboards and review queues.

- **Low-temperature inference (`temperature=0`)**  
  Reduces style variability and improves decision reproducibility for similar frames.

- **Frame-level event summary input**  
  The upstream summary includes counts, risk triggers, and confidence stats, which gives the model enough context without shipping raw box lists.

### Runtime guardrails around the prompt

- If severity is outside `{LOW, MEDIUM, HIGH}`, pipeline coerces to `MEDIUM`.
- If API call fails, pipeline records `MEDIUM` with failure reason rather than dropping the frame.
- High-risk candidate selection remains deterministic and rule-based (person+vehicle and/or low-confidence traffic light), so LLM is used for classification, not candidate discovery.

---

## 5) Example outputs and quality impact across iterations

Examples below are representative of actual failure/success patterns seen during development.

### Scenario A: mixed traffic with pedestrians and vehicles

Event context (abridged): `persons=6, vehicles=9`, no low-confidence lights.

- **v1 output (naive)**  
  `"This appears fairly risky because there are many moving agents."`  
  Issue: no canonical severity token; not machine-safe.

- **v2 output (semi-structured)**  
  ```json
  {"severity":"moderate","reasoning":"Multiple pedestrians and cars in close temporal overlap."}
  ```  
  Issue: `moderate` not in allowed set.

- **final output (production prompt)**  
  ```json
  {"severity":"MEDIUM","reasoning":"Pedestrians and multiple vehicles co-occur in the same frame, increasing interaction risk. No additional uncertainty trigger is present."}
  ```  
  Result: schema-valid, taxonomy-valid, usable without manual correction.

### Scenario B: repeated low-confidence traffic light detections

Event context (abridged): `traffic light min_conf=0.31`, no pedestrians.

- **v1 output**  
  Long narrative with no discrete class.

- **v2 output**  
  JSON returned, but sometimes wrapped in markdown fences.

- **final output**  
  ```json
  {"severity":"MEDIUM","reasoning":"Traffic control state is uncertain due to low-confidence traffic light detection. This warrants caution even without pedestrian-vehicle co-occurrence."}
  ```  
  Result: consistent triage label and concise rationale.

### Scenario C: co-occurrence plus uncertainty trigger

Event context (abridged): pedestrian+vehicle present and low-confidence traffic light in same frame.

- **final output trend**  
  More frequent `HIGH` assignment vs earlier prompts, with rationale explicitly citing **compound risk factors** instead of generic “busy scene” language.

### Net effect on triage quality

- Parsing failure rate dropped from “frequent manual cleanup” to “rare edge-case fallback.”
- Severity labels became analytically usable (`value_counts`, trending, threshold-based alerting).
- Rationale quality shifted from generic scene description to trigger-grounded justifications tied to available detector evidence.

---

## Open items

- Add optional confidence calibration metadata to prompt (camera condition, time-of-day) if available.
- Evaluate whether low-confidence-light-only events should default to `LOW` unless repeated in a temporal window.
- Consider strict JSON schema validation (Pydantic) before row write for additional hardening.
