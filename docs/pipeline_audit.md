# Pipeline Audit Notes

The pipeline combines training, inference, and evaluation. Audit the handoff between stages before comparing results.

## Stage Handoff

- Training should write the checkpoint path and tokenizer settings.
- Inference should record the exact checkpoint and prompt template.
- Evaluation should record the prediction file, gold split, and metric script.

## Failure Checks

- Missing or malformed generated scores.
- Prompt outputs that include rationale text where only scores are expected.
- Score parser failures hidden by default values.
- Evaluation run against a different split than inference.

## Run Bundle

Keep the command lines, stage logs, prediction export, parse failure report, and final metrics together in one external run folder.
