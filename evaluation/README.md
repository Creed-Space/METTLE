# METTLE evaluation data contract

This directory defines the aggregate evaluation contract. It intentionally does
not contain a claimed fairness dataset. A release may report false acceptance or
false rejection only after an independently governed, rights-cleared, held-out
dataset has been collected and reviewed.

Input is JSON Lines governed by `input-schema-v1.json`. Each row contains
`dataset_version`, `subject_class`,
`suite`, `expected_pass`, and `observed_pass`. Allowed subject classes are
`becoming-mind` and `human-assisted`. Optional `cohort` values must be aggregate,
non-identifying labels. Raw prompts, answers, response text, names, contact data,
or stable subject identifiers are forbidden.

Run:

```bash
python3 scripts/evaluate_policy_dataset.py data.jsonl --output aggregate.json
```

The output follows `aggregate-schema-v1.json`. It records counts, false accept
and false reject rates per suite and subject class, and an `insufficient_data`
flag below 30 positive and 30 negative
examples. It is decision input, never proof that a threshold is fair. Threshold
changes require protocol-governance review and a dataset receipt.
