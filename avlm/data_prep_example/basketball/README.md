# Basketball Data-Prep Example (BARD)

This example converts the public [BARD](https://huggingface.co/datasets/GabrieleGiudici/BARD) basketball dataset (14,676 labelled basketball broadcast clips from 60 games, CC BY 4.0) into **HuggingFace conversation JSONL** for train / val / test_seen / test_unseen. Download BARD from Hugging Face to get started (see [Requirements](#requirements)).

The generator implements the 16 templates (14 MCQ, 2 open-ended) of the [categorical evaluation taxonomy](categorical_evaluation/CATEGORICAL_EVAL.MD); the [categorical template examples](categorical_evaluation/TEMPLATE_EXAMPLES_CATEGORICAL.MD) give counts, baselines and one real example per class. The process is deterministic.

## Requirements

| Need | Packages |
|---|---|
| **Generate + split** | Python 3.8+, standard library only |
| **Media probe** (`probe_clip_media.py`) | `decord` (see repo [THIRD_PARTY_NOTICES](../../../THIRD_PARTY_NOTICES)) |
| **Data** | BARD from Hugging Face (~324 GB; the complete dataset is required): `git lfs install && git clone https://huggingface.co/datasets/GabrieleGiudici/BARD raw_basketball_data_input/BARD` (run inside this directory) |

## Quick start

Once the [Requirements](#requirements) above are satisfied:

```bash
cd avlm/data_prep_example/basketball
python prepare_training_data_pipeline.py
```

The first stage probes every labelled clip with `decord`; pass `--media-json <path>` to reuse an existing media table and `--dataset-root <path>` if BARD lives elsewhere. Output lands in `training_basketball_data_output/`: `bard_cat_{train,val,test_seen,test_unseen}.jsonl`, per-split video metadata, and `split_stats.json`.

To have an AI coding agent do this for you, point it at the [prepare-data skill](../../../skills/prepare-data/SKILL.md) and its [basketball guide](../../../skills/prepare-data/references/basketball.md): it checks what is already in place, asks only for what is missing, and runs the pipeline. [ONBOARD_NEW_DATASET.md](../ONBOARD_NEW_DATASET.md) is the general recipe for building such an example from any labelled video dataset.

## Pipeline

The general pipeline is the one shown in the [tennis example](../tennis/README.md#pipeline). Because this example is built from a published dataset, three of its stages do not apply: there is no annotation or clip extraction (the clips and labels are used as released), no LLM distractors (every option comes from the labels), and no post-processing (the splits written by the generator are the final training and evaluation data).

```
raw_basketball_data_input/BARD/dataset_paths.csv + captions/caption.json + <game>/<n>.mp4
      │  probe_clip_media.py             (decord: duration, frames, fps of every labelled clip)
      ▼
training_basketball_data_output/clip_media.json
      │  generate_categorical_mcq_qa.py  (16 templates + data-quality gates + game-level split)
      ▼
training_basketball_data_output/bard_cat_{train,val,test_seen,test_unseen}.jsonl
training_basketball_data_output/bard_cat_<split>_meta_input.jsonl, bard_cat_<split>_meta_input_video_metadata.json, split_stats.json
```

Each clip becomes many training examples — one per template whose ambiguity filter it passes: 175,193 rows from 14,676 clips, about 12 per clip.

## Input format

`dataset_paths.csv` (`;`-separated, one row per labelled clip): `urls` is the clip path relative to the BARD root, `actions` the ordered play-by-play list of action slots, `numerosity` the slot count.

```
urls;actions;numerosity
bkn-vs-det-0022400861/7.mp4;[{'player': '0', 'action': 'Turnover', 'result': None, 'assisted': None, 'other_player': None, 'color': 'black'}, {'player': '8', 'action': 'Steal', 'result': None, 'assisted': None, 'other_player': None, 'color': 'white'}, {'player': '8', 'action': '2PT Shot', 'result': True, 'assisted': True, 'other_player': '45', 'color': 'white'}];3
```

Slot fields: `player` (jersey number), `color` (jersey colour; two per game), `action` (one of nine: `2PT Shot`, `3PT Shot`, `Free Throw`, `Rebound`, `Foul`, `Turnover`, `Steal`, `Block`, `Violation`), `result` (`True`/`False` for shots, else `None`), `assisted` (shots only) and `other_player` (the assister's jersey number). List order is play-by-play order.

`captions/caption.json` holds one BARD-format caption per clip, used verbatim as the `clip_caption` target:

```
A player with jersey number 5 and jersey_color white made a 2PT Shot which result was made and was assisted by other player with jersey number 22
A player with jersey number 13 and jersey_color blue made a Violation
```

## How the generator builds rows

`generate_categorical_mcq_qa.py` walks every clip and applies the 16 templates. Actors are addressed visually — "the player wearing the black number 8 jersey" — because `(colour, number)` identifies the acting player uniquely in almost every action slot, so multi-action clips stay usable. Each template specifies:

- **Ground-truth field(s)** — read from the addressed slot (`action`, `result`, `assisted`, `player`, `color`, `other_player`) or derived from several slots, list order or negative evidence (for example `rebound_side`, `action_order`).
- **An ambiguity filter** — a per-class rule that guarantees exactly one correct answer, such as "the addressed player has exactly one logged action"; everything a filter removes is counted in `split_stats.json`.
- **Option type** — **fixed** (a closed vocabulary such as the three shot types or made/missed), **derived** (sampled distractors such as same-team jersey numbers from the same game, so colour cannot eliminate options) or **open_ended** (`clip_caption` verbatim, or a templated one-sentence description).

For each row the generator: (1) addresses the actor, (2) applies the class filter, (3) reads or derives the correct answer, (4) shuffles the options with a per-row seed, (5) wraps everything in an HF `conversation` with `id`, `class`, `evaluation_type`, `super_category` and `fine_category`, and the clip's relative `video` path. Build-time data-quality gates (a mislabelled game excluded from the identity classes, corrupt rows dropped, per-class totals asserted) are listed in [CATEGORICAL_EVAL.MD → Data-quality gates](categorical_evaluation/CATEGORICAL_EVAL.MD#data-quality-gates).

## Splits

Games are shuffled with `--seed 42` and split **by game, never by clip**, so every row derived from one clip stays in one split: **train** / **val** (a disjoint 5% of rows from the same clip pool, so it tracks fitting rather than generalization) / **test_seen** (a held-out 10% of clips per seen game) / **test_unseen** (nine whole held-out games; no game shared with any other split).

**Duration confound.** BARD cuts clips on play-by-play boundaries, so clip length tracks event count. `split_stats.json` therefore reports, per class and split, chance, majority-class and a **duration-only** baseline to read every model score against.

## Output format

One JSON object per line; each record is one (video, question) pair in HF conversation form. `class` is the template name, `evaluation_type` routes MCQ vs LLM-judge scoring, and `super_category` / `fine_category` come from the taxonomy:

```json
{
  "id": "mil-vs-dal-0022400865_38_player_action_green34",
  "class": "player_action",
  "evaluation_type": "mcq",
  "super_category": "Fine-Grained Visual Recognition",
  "fine_category": "Action Attribution",
  "conversation": [
    {
      "role": "user",
      "content": [
        {"type": "video", "path": "mil-vs-dal-0022400865/38.mp4"},
        {"type": "text", "text": "What does the player wearing the green number 34 jersey do in this video?\n(A) a three point shot\n(B) a turnover\n(C) a violation\n(D) a free throw"}
      ]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "(D) a free throw"}]
    }
  ]
}
```

Open-ended rows omit the `(A)/(B)/...` options; the assistant text is the caption or the templated description directly. Options are lettered `(A)`–`(E)`; answer distributions are allowed to be imbalanced.

Per split the generator writes:

| File | Contents |
|---|---|
| `bard_cat_<split>.jsonl` | HF conversation rows |
| `bard_cat_<split>_meta_input.jsonl` | one `{id, video}` row per referenced clip |
| `bard_cat_<split>_meta_input_video_metadata.json` | `{path: [duration_s, frames, fps]}` per referenced clip, in the format of `avlm/utils/build_video_metadata.py` |
| `split_stats.json` | per-split, per-class counts, answer mix, chance / majority / duration-only baselines, exclusion counts |

## Documents

- [Categorical evaluation taxonomy](categorical_evaluation/CATEGORICAL_EVAL.MD) — super / fine categories, per-class ground truth, data-quality gates
- [Categorical template examples](categorical_evaluation/TEMPLATE_EXAMPLES_CATEGORICAL.MD) — per-class counts, answer mix, baselines, one real example per class
