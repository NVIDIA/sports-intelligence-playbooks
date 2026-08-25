# Tennis rules MCQ — trigger map

How we decide **which rules questions** attach to a tennis point, and how those become MCQs.

**Source of truth for generation:** `configs/rules/tennis_rules_trigger_map.json`  
**Question text / options:** `configs/rules/tennis_rules_question_bank.json`  
**Code:** `tennis_rules_knowledge.py`

| Artifact | Path |
|---|---|
| Question bank | `configs/rules/tennis_rules_question_bank.json` |
| Trigger map | `configs/rules/tennis_rules_trigger_map.json` |
| Selector | `tennis_rules_knowledge.py` |
| Templates | `rules_knowledge_contextual` (+ `_secondary`) in `configs/generation/tennis_categorical_eval_config.json` |
| Paraphrase bank | `configs/paraphrases/rules_knowledge_paraphrase_bank.json` |

---

## What these MCQs are

- The **video** is the point clip (so the situation feels related).
- The **answer** is always the **general tennis rule**, not “what happened in this clip.”
- Example: on an Ace point we may ask “What is an ace?” — not “Was this an ace?”

---

## End-to-end flow (one point)

```
Point annotation
       │
       ▼
1. Match   For each of the 27 contextual questions, check triggers
           (any / all predicates on this point’s fields).
       │
       ▼
2. Select  Keep at most max_per_point = 2 questions
           (diversity: different families when possible).
       │
       ▼
3. Emit    For each selected question:
           • pick a stem from the bank
           • use the bank’s fixed A/B/C/D options + correct answer
           • attach the point video
       │
       ▼
4. Later   Split-aware paraphrase of stems/options (train vs eval pools)
```

Two generator templates call the same selector:

| Template | Slot | When it emits |
|---|---:|---|
| `rules_knowledge_contextual` | 0 (primary) | If ≥1 question matched |
| `rules_knowledge_contextual_secondary` | 1 | If ≥2 questions matched |

Both write HF class `rules_knowledge_contextual`.

---

## Annotation fields used for triggers

| Field | Used for |
|---|---|
| `how_point_ended` | Ace, Double Fault, Winner, Forced/Unforced Error, … (label before `:`) |
| `num_serving_attempts_until_successful` | First vs second serve |
| `loser_point_ended` | net / out / no-contact |
| `score_before`, `score_after` | Love, deuce, ad, break/game point, game won |
| `winner_shot_type` / `loser_shot_type` | Volley (weak) |
| caption / description text | Optional weak hints only |

---

## Selecting among matches (keep it simple)

Many points match **more than two** questions. We do **not** emit all of them.

1. Group matches into **families**: `outcome`, `serve`, `score`, `geometry`, `rally`
2. Prefer **different families** for the two slots (so we don’t pick two serve questions)
3. Within a family, rotate among the better-priority matches so the same few IDs don’t always win
4. Cap: **`max_per_point = 2`**

| Family | Question IDs |
|---|---|
| `outcome` | 3, 6, 13, 26 |
| `serve` | 1, 2, 12, 30, 31, 36 |
| `score` | 18–25, 37, 38 |
| `geometry` | 7, 15, 16, 39 |
| `rally` | 27, 29, 35 |

**Generic-only** IDs (4, 5, 8–11, 14, 17, 28, 32–34, 40) never emit in v1 — no reliable annotation field.

---

## Worked example

### Point annotation (simplified)

```text
how_point_ended:                        Ace
num_serving_attempts_until_successful:  1
score_before:                           "..." {4-3, 30-15}
score_after:                            "..." {4-3, 40-15}
loser_point_ended:                      (empty)
winner_shot_type:                       serve
```

### Step 1 — which questions match?

| ID | Topic | Why it matches |
|---:|---|---|
| 6 | Definition of an ace | outcome = Ace |
| 1 | Legal serve landing | serve attempt / Ace |
| 12 | Serve before bounce | same serve context |
| 36 | Service box lines | same serve context |
| 18 | Scoring sequence | parseable scores |
| 31 | Alternating serve sides | parseable score, not 0–0 |
| … | (others) | may also match depending on fields |

(Double-fault / winner / net / volley questions do **not** match this Ace point.)

### Step 2 — pick up to 2 (diversity)

Example selection (families differ):

| Slot | ID | Family | Topic |
|---:|---:|---|---|
| 0 | **6** | outcome | Definition of an ace |
| 1 | **1** | serve | Legal serve landing location |

(Another point id might rotate to Q6 + Q18, or Q12 + Q23, etc.)

### Step 3 — build the MCQ from the bank

For Q6 the bank supplies stems + fixed options. One emitted sample:

**Question (stem):**  
Which of the following correctly defines an ace?

**(A)** Any serve that the receiver touches but fails to return in court  
**(B)** A serve that lands in and is not touched by the receiver  
**(C)** A non-serve groundstroke that lands in and is not reached by the opponent  
**(D)** A second serve that lands in after a fault on the first serve  

**Correct:** (B)

Video path = this point’s clip. Secondary slot does the same for Q1 (different stem/options from the bank).

**Important:** we are **not** asking “Did the server hit an ace here?” The clip is context; the gold answer is the rule definition.

---

## Trigger map

Legend:
- **Mode**: `contextual` = can emit · `generic-only` = skipped in v1
- **Confidence**: `high` = structured field is enough · `medium` = usually OK · `low` = weak / skip
- **Fire when**: annotation predicate that must hold

| ID | Topic | Mode | Confidence | Fire when (annotation predicate) | Notes |
|---:|---|---|---|---|---|
| 1 | Legal serve landing location | contextual | high | Any point with a resolved serve attempt (`num_serving_attempts_until_successful` ≥ 1) **or** outcome in {Ace, Double Fault} | Broad serve-context question |
| 2 | First-serve fault | contextual | high | `num_serving_attempts_until_successful` = 2 **or** outcome = Double Fault | Point involved a missed first serve |
| 3 | Double fault | contextual | high | outcome = Double Fault | |
| 4 | Let serve that lands in | generic-only | low | *(no reliable field)* | Only if future `let`/`serve_let` label exists |
| 5 | Serve touches net then out | generic-only | low | *(no reliable field)* | Same as Q4 |
| 6 | Definition of an ace | contextual | high | outcome = Ace | Strong thematic match |
| 7 | Ball landing on a line | contextual | medium | outcome in {Winner, Forced Error, Unforced Error} **or** `loser_point_ended` indicates out/net/no-contact | Related to in/out judgment; not proof the ball hit a line |
| 8 | Double bounce | generic-only | low | *(rarely annotated)* | Skip unless Violation / description explicitly mentions double bounce |
| 9 | Touching the net while ball in play | generic-only | low | outcome = Violation only if description mentions net touch | Otherwise skip |
| 10 | Ball striking a player | generic-only | low | *(rarely annotated)* | Optional text match on caption/description |
| 11 | Foot fault | generic-only | low | *(not in structured fields)* | Skip in v1 |
| 12 | Serve must be struck before bouncing | contextual | medium | Any serve-related point (same as Q1) | General serve rule; weak situational specificity |
| 13 | Definition of a winner | contextual | high | outcome = Winner | Strong thematic match |
| 14 | Net cord during a rally | generic-only | low | *(not structured)* | Skip unless caption mentions net cord / luck / dribbled over |
| 15 | Shot into the net during a rally | contextual | high | `loser_point_ended` ≈ net **or** description/caption clearly indicates into the net | Prefer structured `loser_point_ended` |
| 16 | Singles court boundaries | contextual | medium | `loser_point_ended` ≈ out **or** text hints wide/sideline/alley | Thematic for wide/out balls |
| 17 | Serve strikes receiver before bounce | generic-only | low | *(rare)* | Optional: Ace + caption says hit body; else skip |
| 18 | Point scoring sequence | contextual | medium | Any point with parseable `score_before` and `score_after` | Always thematically related to scoring |
| 19 | Deuce | contextual | high | `score_before` is deuce (40–40 / “deuce”) **or** `score_after` becomes deuce | |
| 20 | Advantage after deuce | contextual | high | `score_before` is deuce **and** `score_after` is advantage | |
| 21 | Losing advantage (back to deuce) | contextual | high | `score_before` is advantage **and** `score_after` is deuce | |
| 22 | Winning from advantage | contextual | high | `score_before` is advantage **and** game is won (`score_after` shows game/set advance) | |
| 23 | Meaning of “Love” | contextual | medium | `score_before` or `score_after` contains Love / 0 in the point score | |
| 24 | Break point | contextual | high | `score_before` is a break-point state for the receiver | Receiver can win game on next point |
| 25 | Hold vs break | contextual | high | End of a service game where the **server** won the game | Game-ending points only |
| 26 | Forced vs unforced error | contextual | high | outcome in {Forced Error, Unforced Error} | Asks the definition, not which one occurred |
| 27 | One bounce only before return | contextual | medium | Rally outcomes: Winner / Forced Error / Unforced Error | |
| 28 | Racket may strike the ball only once | generic-only | low | *(not annotated)* | Skip in v1 |
| 29 | Volley definition and legality | contextual | medium | `winner_shot_type` or `loser_shot_type` indicates volley **or** caption mentions volley | |
| 30 | Serving from the correct half | contextual | medium | Parseable `score_before` **and** score is game start (0–0) | Deuce-court vs ad-court |
| 31 | Alternating serve sides within a game | contextual | medium | Parseable scores **and** score is **not** game start | Companion to Q30 |
| 32 | Receiver hit by legal return before bouncing | generic-only | low | *(rare)* | Skip in v1 unless body-hit labeled |
| 33 | Ball goes through the net | generic-only | low | *(not annotated)* | Skip in v1 |
| 34 | Hindrance / external disturbance let | generic-only | low | outcome = Let **and** not a serve-let (if distinguishable) | Keep conservative |
| 35 | Returning after a legal bounce | contextual | medium | Rally outcomes (same as Q27) | |
| 36 | Service box lines on serve | contextual | high | Serve-related points (same as Q1) | Pairs well with ace / DF / second serve |
| 37 | Game point for the server | contextual | high | `score_before` is server game point (e.g. 40–30, 40–15, 40–0, AD server) | |
| 38 | Meaning of a break | contextual | high | Game-ending point where **receiver** wins the service game | Companion to Q25 |
| 39 | Ball lands beyond the baseline | contextual | high | `loser_point_ended` ≈ out **or** text says long / beyond baseline | Prefer structured out label |
| 40 | No-ad scoring | generic-only | low | *(format not in point labels)* | Skip in v1 |

### Emit set (v1)

**Contextual (27):** 1, 2, 3, 6, 7, 12, 13, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 35, 36, 37, 38, 39  

**Generic-only / skipped (13):** 4, 5, 8, 9, 10, 11, 14, 17, 28, 32, 33, 34, 40
