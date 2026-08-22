<!-- markdownlint-disable MD013 -->

# Documentation guide

## Executive summary

Turn detection is an asymmetric safety problem: interrupting an unfinished speaker costs more than waiting briefly.
The data pipeline retains silence, annotation provenance, and endpoint alignment rather than replacing them with convenient shortcuts.
Hinglish-focused augmentation varies fillers, pauses, speaking rate, pitch, noise, and volume without changing labels.
Matched experiments isolate data, pooling, pause policy, encoder adaptation, hard mining, and text semantics.
The analysis considers false-complete rate, hard-case slices, latency, and model size alongside aggregate F1.
Failed hypotheses and evidence limits remain in the record, so later decisions follow the results.
Safety finalists ran across three seeds and used validation-only calibration. E4 won, bringing held-out FCR below 10% with a recall trade-off.

## Core reading path

1. [Data preparation approach](01_data_preparation_approach.md): raw-data findings, evidence gaps, cleaning, targeted augmentation, hard-example mining, and split rationale.
2. [Experiment plan](02_experiment_plan.md): research questions, controlled comparisons, pre-registered hypotheses, metrics, success criteria, and validity threats.
3. [Ablation insights](03_ablation_insights.md): held-out results, filler/pause error analysis, qualitative cases, failed assumptions, and evidence-driven decisions.
4. [Full approach](04_full_approach.md): complete problem-to-solution narrative covering data, modeling, experimentation, final system, and limitations.
5. [Zero-to-mastery tutorial](05_zero_to_mastery_tutorial.md): beginner-first explanation of the problem, data, model, training, experiments, and deployment.

Read in order to follow the path from data findings to experimental decisions. Start with the full approach when you need one consolidated overview.

## Executable notebooks

- [Data preparation](../notebooks/01_data_preparation.ipynb): statistics, pause analysis, augmentation previews, hard examples, and split audits.
- [Experiment design](../notebooks/02_experiment_design.ipynb): comparator controls, hypotheses, success rules, feasibility gates, and logging contract.
- [Ablations and results](../notebooks/03_ablations_and_results.ipynb): artifact validation, paired effects, training curves, slice analysis, and embedded qualitative audio.

## Supporting references

- [Implementation architecture](codebase/ARCHITECTURE.md): traced data, training, model, experiment, inference, and Gradio flows.
- [Codebase notes](codebase/STRUCTURE.md): stack, structure, conventions, integrations, tests, and review findings.
- Mermaid sources: [system architecture](diagrams/system-architecture.mmd), [data lifecycle](diagrams/data-lifecycle.mmd), [module dependencies](diagrams/module-dependencies.mmd), and [inference sequence](diagrams/inference-sequence.mmd).
- [Codebase guide](codebase_guide.md): module ownership, data flow, contracts, and extension points.
- [Generated data exploration](data_exploration.md): reproducible statistics from local decoded artifacts.
- [Submission checkpoint](submission_checkpoint.md): bundled demo checkpoint, training-budget distinction, and deployment caveats.
- [Safety finalist summary](generated/safety_v1_summary.md): nine-run selection, calibrated operating point, and one frozen held-out evaluation.
- `generated/`: machine-rendered experiment reports. The numbered documents above contain the curated conclusions.

## Evidence boundary

The documents distinguish publisher metadata, partial Dataset Viewer statistics, and the decoded local subset. Hindi, filler, and pause slices are Hinglish proxies because the source data has no verified code-switch or speaker-identity labels. The project does not claim a true Hinglish benchmark or speaker-disjoint evaluation.
