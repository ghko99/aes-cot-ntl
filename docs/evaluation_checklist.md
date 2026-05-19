# Evaluation Checklist

This project explores AES scoring with chain-of-thought style prompting and number-token loss variants. Keep evaluation records explicit.

## Before Evaluation

- Confirm the checkpoint or adapter path.
- Record prompt template version and whether rationales are requested.
- Verify score parsing handles every rubric field.
- Use the same held-out split when comparing loss variants.

## Metrics To Save

- Overall and per-rubric QWK.
- MAE and RMSE.
- Exact score accuracy for parser diagnostics.
- Parse failure count and examples.

## Result Notes

Report the model path, adapter revision, decoding settings, split name, and metric script. Keep raw generations so score parsing issues can be audited later.
