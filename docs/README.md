<!-- markdownlint-disable MD013 -->

# Documentation Guide

## Executive summary

This project treats turn detection as an asymmetric safety problem: interrupting an unfinished speaker costs more than waiting briefly.  
Data work preserves silence, annotation provenance, and endpoint alignment instead of reducing speech to convenient shortcuts.  
Hinglish-focused augmentation tests fillers, pauses, speaking rate, pitch, noise, and volume while keeping labels honest.  
Experiments isolate data, pooling, pause policy, encoder adaptation, hard mining, and text semantics under matched controls.  
Conclusions use false-complete rate, hard-case slices, latency, and model size alongside aggregate F1.  
Results include failed hypotheses and evidence limits, so next decisions follow what experiments showed rather than expected.

## Core reading path

1. [Data preparation approach](01_data_preparation_approach.md) — raw-data findings, evidence gaps, cleaning, targeted augmentation, hard-example mining, and split rationale.
2. [Experiment plan](02_experiment_plan.md) — research questions, controlled comparisons, pre-registered hypotheses, metrics, success criteria, and validity threats.
3. [Ablation insights](03_ablation_insights.md) — held-out results, filler/pause error analysis, qualitative cases, failed assumptions, and evidence-driven decisions.
4. [Full approach](04_full_approach.md) — complete problem-to-solution narrative covering data, modeling, experimentation, final system, and honest limitations.

Read in order to follow reasoning from observed data problems to experiment-driven decisions. Read the full approach first when a single consolidated overview is needed.

## Executable notebooks

- [Data preparation](../notebooks/01_data_preparation.ipynb) — statistics, pause analysis, augmentation previews, hard examples, and split audits.
- [Experiment design](../notebooks/02_experiment_design.ipynb) — comparator controls, hypotheses, success rules, feasibility gates, and logging contract.
- [Ablations and results](../notebooks/03_ablations_and_results.ipynb) — artifact validation, paired effects, training curves, slice analysis, and embedded qualitative audio.

## Supporting references

- [Codebase guide](codebase_guide.md) — module ownership, data flow, contracts, and extension points.
- [Generated data exploration](data_exploration.md) — reproducible statistics from local decoded artifacts.
- [Submission checkpoint](submission_checkpoint.md) — bundled demo checkpoint, training-budget distinction, and deployment caveats.
- `generated/` — machine-rendered experiment reports; curated conclusions remain in numbered documents above.

## Evidence boundary

Claims distinguish publisher metadata, partial Dataset Viewer statistics, and the decoded local subset. Hindi, filler, and pause slices are Hinglish proxies because source data has no verified code-switch or speaker identity labels. Documents therefore avoid claiming a true Hinglish benchmark or speaker-disjoint evaluation.
