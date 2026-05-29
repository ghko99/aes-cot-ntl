# Score Parsing Notes

Score parsing is a fragile boundary between generation and evaluation. Keep it explicit for CoT and NTL experiments.

## Parser Checks

- Confirm every rubric field is present.
- Reject scores outside the valid range.
- Track responses with rationale text mixed into score fields.
- Count missing, duplicated, and malformed scores.
- Save parse failures separately from valid predictions.

## Evaluation Handling

Do not silently replace parse failures with default scores. Report parse failure count next to aggregate metrics so prompt or model changes can be compared fairly.

## Debug Samples

Keep a small synthetic set of expected outputs and malformed outputs for parser regression checks.
