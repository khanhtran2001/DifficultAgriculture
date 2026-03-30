---
name: scientific-analysis-workflow
description: "Run rigorous scientific analysis for ML experiments in this repository. Use for hypothesis-driven experiment design, controlled comparisons, reproducibility checks, and result interpretation for MinneApple/YOLO pipelines."
argument-hint: "analysis goal, target metric, comparison scope"
user-invocable: true
disable-model-invocation: false
---

# Scientific Analysis Workflow

## What This Skill Produces

A reproducible analysis package for a given task:
- clear hypothesis and success criteria
- controlled experiment plan
- run artifacts with frozen config and outputs
- statistical and practical interpretation
- decision recommendation and next experiment

## When to Use

Use this skill when you need one or more of these outcomes:
- compare baseline vs variant models or augmentations
- validate whether a change improves quality on target metrics
- analyze dataset quality and error patterns before training changes
- prepare a reproducible analysis summary for collaborators

Typical trigger words:
- scientific analysis, hypothesis testing, controlled comparison
- ablation, significance, confidence interval, reproducibility
- baseline vs new method, scoring analysis

## Inputs To Request

Collect these inputs first:
- task objective and claim to test
- target metric(s): default primary metric is mAP; optionally track AP50, AP75, AP small/medium/large
- dataset split policy and random seed policy
- compute budget and run count
- acceptance threshold for practical improvement

If inputs are missing, ask only for blockers and use defaults for non-blockers.

## Procedure

1. Frame the hypothesis
- Convert the request into testable form:
  - H0: no improvement over baseline.
  - H1: improvement over baseline on chosen metric(s).
- Define minimum practical effect size (for example, +1.0 mAP).

2. Lock reproducibility controls
- Fix experiment config and seed list before runs.
- Preserve frozen config artifacts for each run.
- Keep train/val/test split and evaluation protocol unchanged.

3. Build a controlled comparison plan
- Start with a baseline condition.
- Add one change per variant (single-factor changes first).
- Keep all non-target variables constant (epochs, image size, thresholds, data split).
- Plan at least 3 seeds when budget allows.

4. Execute and collect outputs
- Run baseline and variants with identical evaluation paths.
- Save run outputs to stable result directories.
- Capture training metrics, evaluation metrics, and prediction artifacts used for scoring.

5. Validate run integrity
- Check that each run finished without obvious failure modes:
  - corrupted or missing outputs
  - mismatched config fields
  - inconsistent thresholds or split leakage
- Reject and rerun invalid runs rather than averaging bad data.

6. Analyze with uncertainty
- Aggregate by condition across seeds: mean, std, and deltas vs baseline.
- Prefer confidence intervals or bootstrap intervals over single-point claims.
- Report both statistical evidence and practical effect size.

7. Perform error-focused diagnostics
- Slice metrics by object scale, scene density, and confidence thresholds.
- Inspect representative false positives/false negatives.
- Determine whether gains come from true generalization or threshold artifacts.

8. Decide and recommend
- Accept change only if:
  - metric improvement passes practical threshold, and
  - uncertainty does not contradict the claimed direction (balanced evidence standard).
- Produce one clear recommendation:
  - adopt, reject, or run follow-up ablation.

## Decision Points And Branching Logic

- If variance across seeds is high:
  - increase seed count before concluding.
- If baseline underperforms historical runs:
  - stop and validate environment/config parity first.
- If gains appear only at a narrow threshold:
  - classify as calibration effect; require robustness checks.
- If augmentation helps one slice but hurts another:
  - propose conditional strategy or targeted augmentation.
- If compute is limited:
  - run a two-stage plan:
    - Stage A: quick screen with fewer epochs.
    - Stage B: confirm only top candidates at full budget.

## Quality Gates (Completion Checks)

Mark analysis complete only when all checks pass:
- hypothesis and success criteria are explicit
- baseline and variant configs are frozen and archived
- at least one controlled baseline-vs-variant comparison is executed
- uncertainty-aware summary is provided (not single-run only)
- key failure cases are inspected with evidence
- final recommendation includes rationale and risks

Balanced evidence default:
- Prefer >= 3 seeds when budget allows.
- Require directional consistency across most seeds before adoption.
- If uncertainty overlaps zero effect, recommend follow-up runs instead of adoption.

## Output Template

Use this structure for the final report:
1. Objective and hypothesis
2. Experimental setup and controls
3. Results table (per seed and aggregate)
4. Uncertainty and effect-size interpretation
5. Error analysis findings
6. Recommendation and next experiment

## Repository Notes

For this repository, prefer alignment with existing experiment flow:
- Step 1: dataset validation
- Step 2: baseline train/evaluate
- Step 3: scoring/predictions
- Step 4+: augmentation or variant evaluation

Keep result handling consistent with current overwrite semantics and reproducibility expectations.
