# Onboard a labelled video dataset as MCQ + QA data

Deliverables under `avlm/data_prep_example/<name>/`: a design note agreed with the user, the
processing scripts, and the training-ready splits (HuggingFace conversation JSONL with per-split
video metadata and split statistics). Answers come from the dataset's labels; LLM-assisted
generation is out of scope here. Conventions, file layout and output formats live in the
reference implementations below: read the closest one before writing code.

## Understand the data first

- The sample is what the label attaches to. Cut only when you must; then use a fixed-length
  window centred on the event so clip length carries no label signal.
- Address actors only by attributes visible in the video, never by names.
- Trust list order as temporal order only where corroborated.
- Find the source grouping (game, match, recording), the label defects, and any shipped text
  usable as open-ended targets. Test each source against domain invariants (for example, the
  two sides of one event must differ in team attribute); a source that fails is a defect to gate.
- Measure the media (duration, fps, audio), with a probe or an existing media table, and the
  duration confound: a length-only classifier
  fit on training sources and scored on held-out ones.
- Reuse an upstream split when it is by source; if the shipped test split is label-withheld,
  use the upstream val as test.

## Work with the user at three points

1. **Intake.** Show the schema as you understand it, then ask what the labels cannot tell you.
   Domain: which events are visually verifiable within a sample, which fields are trustworthy,
   how each event and actor should be phrased, what counts as a source, which event pairs have
   a fixed order, known defects. Policy: purpose, capabilities to cover, allowed ground truth,
   audio at train and eval, open-ended target style, minimum support and imbalance policy, row
   budget, comparability with existing splits. Offer the reference's choice as the default.
2. **Menu cut.** Enumerate every buildable candidate before curating: one per label field (what
   is X); a targeted variant wherever a sample holds several actors or events, addressed by
   visible attributes; derived candidates from relations between fields (order, co-occurrence,
   negative evidence, same-side tests) only where corroborated; identity candidates wherever
   actors are addressable; open-ended candidates from shipped text and from templated renders
   of the fields. Merge candidates that share a question form and differ only in the asked
   event type into one class with the type in the wording. Attach rows per split, answer mix and
   baselines to each, and let the user keep, drop or merge.
3. **Design confirmation.** The curated taxonomy from the surviving candidates, each meeting the
   checklist below, grouped into super and fine categories, plus the splits and gates.

Ask in plain language, one round each.

## What every class needs

Each surviving candidate must satisfy:

| Item | Requirement |
|---|---|
| ground truth | a label field of the addressed event, or a derived rule with the reason it can be trusted |
| ambiguity filter | a condition that leaves exactly one correct answer |
| options | a fixed vocabulary, or distractors that side information cannot eliminate; open-ended for text targets |
| wording | one question sentence in the shared vocabulary of the set |
| numbers | rows per split after the filter, answer mix, chance, majority and duration-only baselines |
| example | one real row |

Cover MCQ and open-ended QA. Record rejected candidates with the reason. Write the agreed
design, hierarchy included, to `DESIGN.md`.

## Build principles

- Split by source, never by clip; all rows of one clip stay in one split. Default: whole
  held-out sources for `test_unseen`, held-out clips of seen sources for `test_seen`, and `val` as
  a row sample of the train clips, so it tracks fitting rather than generalization.
- Gate defects (corrupt rows, inconsistently labelled sources) and count every exclusion.
- Shuffle options per row with a seed from the row id; use deterministic ids; assert per-class
  totals against the design, so regeneration is byte-identical and drift stops the build.
- Fail loudly on unparseable rows or missing files; never skip silently.
- Only the fixed row keys (`id`, `class`, `evaluation_type`, `super_category`,
  `fine_category`, video path) survive inference: every distinction must live in `class`.

## Verify independently

- Totals match the design; unseen sources appear in no other split; test_seen clips are absent
  from train and val.
- Regenerating into a scratch directory is byte-identical.
- Answers re-derived from the raw labels by separate code agree; every video path resolves;
  answer letters are near-uniform per class.
- Watch one clip per class and confirm the question matches what is visible.

Report rows per split and class, the baselines, and what was excluded and why.

## Reference implementations

- `avlm/data_prep_example/basketball/` — clip-level source; MCQ + QA, derived classes, source
  gate, per-row shuffling, duration baseline.
- `avlm/data_prep_example/tennis/` — segments cut from full matches; config-driven templates.
