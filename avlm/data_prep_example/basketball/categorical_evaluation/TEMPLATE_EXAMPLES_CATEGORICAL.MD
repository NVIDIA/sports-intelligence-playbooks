# BARD categorical template examples

The 15 class families of the BARD basketball example, with real counts and one real example each,
generated from `dataset_paths.csv` (14,676 clips, 23,523 action slots, 60 games) and
`captions/caption.json`.

## Schema

Follows the tennis set (`tennis_mcq_qa_train.jsonl`): `class` is the **question type** and `evaluation_type`
is `mcq` or `open_ended`, so both live in one file. `class`, `evaluation_type`, `super_category` and
`fine_category` are propagated into `predictions.jsonl` by `avlm/inference/common/run_inference.py`. Options
are lettered `(A)`, `(B)`, …, and **answer distributions are allowed to be imbalanced**.

```json
{"id": "...", "class": "shot_type", "evaluation_type": "mcq",
 "super_category": "Fine-Grained Visual Recognition", "fine_category": "Shot Mechanics",
 "conversation": [{"role": "user", "content": [{"type": "video", "path": "<game>/<n>.mp4"},
                                               {"type": "text", "text": "<question>"}]},
                  {"role": "assistant", "content": [{"type": "text", "text": "(B) ..."}]}]}
```

Two facts that shape the whole set:

- **A clip is the sample.** BARD is clip classification, so nothing is cut and `video_root` is the BARD tree itself.
- **Actors are addressable.** `(jersey colour, jersey number)` uniquely identifies the actor in **20,720 of 22,036**
  action slots (94%), so a question can name a player visually. That makes multi-action clips usable rather than
  discardable, and removes most of the ambiguity filters a clip-level question would need.

## Summary

| class | type | rows | opts | chance | majority | duration-only | super_category |
|---|---|---:|---:|---:|---:|---:|---|
| `player_action` | mcq | 20,720 | 4 of 9 | 25.0% | n/a | n/a | Fine-Grained Visual Recognition |
| `shot_type` | mcq | 11,664 | 3 | 33.3% | 44.0% | 49.8% | Fine-Grained Visual Recognition |
| `shot_type_targeted` | mcq | 12,672 | 3 | 33.3% | 45.1% | 50.5% | Fine-Grained Visual Recognition |
| `shot_result` | mcq | 11,087 | 2 | 50.0% | 56.7% | 55.4% | Fine-Grained Visual Recognition |
| `shot_result_targeted` | mcq | 12,672 | 2 | 50.0% | 53.9% | 54.8% | Fine-Grained Visual Recognition |
| `shot_assisted` | mcq | 4,510 | 2 | 50.0% | 69.7% | 69.7% | Fine-Grained Visual Recognition |
| `rebound_side` | mcq | 3,599 | 2 | 50.0% | 87.0% | 86.4% | Fine-Grained Visual Recognition |
| `block_present` | mcq | 4,249 | 2 | 50.0% | 90.8% | 90.5% | Fine-Grained Visual Recognition |
| `possession_loss` | mcq | 2,267 | 3 | 33.3% | 63.6% | 63.1% | Match Structure & Reasoning |
| `action_order` | mcq | 6,127 | 2–5 | 46.0% | 39.3% | 40.5% | Match Structure & Reasoning |
| `actor_jersey_number` | mcq | 21,416 | 4 | 25.0% | n/a | n/a | Player Identification |
| `actor_team_colour` | mcq | 21,416 | 2 | 50.0% | 40.3% | 39.3% | Player Identification |
| `assister_jersey_number` | mcq | 3,198 | 4 | 25.0% | n/a | n/a | Player Identification |
| `clip_caption` | open | 14,676 | – | – | – | – | Commentary & Reporting |
| `player_action_description` | open | 20,720 | – | – | – | – | Commentary & Reporting |

**Totals: 135,597 MCQ rows + 35,396 open-ended rows across 15 classes.**
`duration-only` is a 10-bin max-likelihood classifier on clip length alone, fit on 51 games and scored on 9
held-out games — compare it against the majority column, not against chance.

Every class below states its **Label source**: DIRECT means the answer is read off a BARD field of the
addressed action slot (any filter is a within-clip aggregation of the same fields); DERIVED means the answer
is computed from several fields, list order, or negative evidence — with the reason it can be trusted.

## Fine-Grained Visual Recognition

### `player_action`

- **evaluation_type** `mcq` · **fine_category** Action Attribution
- **20,720 rows** · 4 options out of 9 actions
- answer mix: `a two point shot` 24.7%; `a three point shot` 20.9%; `a rebound` 20.4%; `a free throw` 12.1% …
- **Label source:** DIRECT — the answer is the slot's `action` field; the actor address is the same slot's `color` + `player`. Distractors (3 sampled from the other 8 types) cannot be a second correct answer because the uniqueness filter leaves the named player exactly one logged action.
- Targeted at one actor, so multi-action clips are usable instead of discarded. 12,221 of the rows come from multi-action clips, where the action must actually be attributed to the named player.

```
video: mil-vs-dal-0022400865/38.mp4
Q: What does the player wearing the green number 34 jersey do in this video?
   (A) a turnover
   (B) a violation
   (C) a three point shot
   (D) a free throw
A: (D) a free throw
```

### `shot_type`

- **evaluation_type** `mcq` · **fine_category** Shot Mechanics
- **11,664 rows** · 3 options · chance 33.3% · majority 44.0% · duration-only 49.8%
- answer mix: `a two point shot` 44.0%; `a three point shot` 34.3%; `a free throw` 21.7%
- **Label source:** DIRECT — the `action` field of the clip's shot slots; the same-type filter is a within-clip aggregation of that field, no inference.
- Clip-level. Requires every shot attempt in the clip to be the same type; 381 mixed-type clips are excluded.

```
video: gsw-vs-nyk-0022400885/224.mp4
Q: What type of shot attempt is shown in this video?
   (A) a three point shot
   (B) a free throw
   (C) a two point shot
A: (C) a two point shot
```

### `shot_type_targeted`

- **evaluation_type** `mcq` · **fine_category** Shot Mechanics
- **12,672 rows** · 3 options · chance 33.3% · majority 45.1% · duration-only 50.5%
- answer mix: `a two point shot` 45.1%; `a three point shot` 35.1%; `a free throw` 19.8%
- **Label source:** DIRECT — the addressed slot's `action` field, addressed by its own `color` + `player`. Unambiguous iff the addressed player attempts exactly ONE shot in the clip; their non-shot actions (rebound, foul, …) don't matter, so the filter is looser than `player_action`'s full uniqueness.
- Player-targeted variant. Needs no same-type filter, so it recovers the 381 mixed-type clips and covers every shot whose shooter takes exactly one shot in the clip (12,672 of 13,047 slots; the 375 dropped are actors with two or more shots, mostly free-throw pairs).

```
video: phi-vs-uta-0022400433/312.mp4
Q: What type of shot does the player wearing the white number 14 jersey attempt in this video?
   (A) a three point shot
   (B) a two point shot
   (C) a free throw
A: (A) a three point shot
```

### `shot_result`

- **evaluation_type** `mcq` · **fine_category** Shot Outcome
- **11,087 rows** · 2 options · chance 50.0% · majority 56.7% · duration-only 55.4%
- answer mix: `the shot is made` 56.7%; `the shot is missed` 43.3%
- **Label source:** DIRECT — the shot slot's boolean `result` field; the single-shot filter is a within-clip count.
- Clip-level, field goals + free throws: every clip with exactly one shot attempt of any type (multi-shot clips are covered by the targeted variant). Folding free throws in brings the 77.6%-made FT prior, which is what moves the majority to 56.7%.

```
video: det-vs-gsw-0022400919/343.mp4
Q: The video shows a single shot attempt. Does the shot go in?
   (A) the shot is missed
   (B) the shot is made
A: (A) the shot is missed
```

### `shot_result_targeted`

- **evaluation_type** `mcq` · **fine_category** Shot Outcome
- **12,672 rows** · 2 options · chance 50.0% · majority 53.9% · duration-only 54.8%
- answer mix: `the shot is made` 53.9%; `the shot is missed` 46.1%
- **Label source:** DIRECT — the addressed slot's boolean `result` field; unambiguous iff the addressed player attempts exactly ONE shot in the clip, as in `shot_type_targeted`.
- Player-targeted variant, field goals + free throws; covers every shot whose shooter takes exactly one shot in the clip (12,672 of 13,047 slots — the same row set as `shot_type_targeted`), including clips holding several shots by different players.

```
video: mil-vs-atl-0022400884/43.mp4
Q: Does the shot taken by the player wearing the red number 11 jersey go in?
   (A) the shot is missed
   (B) the shot is made
A: (B) the shot is made
```

### `shot_assisted`

- **evaluation_type** `mcq` · **fine_category** Play Construction
- **4,510 rows** · 2 options · chance 50.0% · majority 69.7% · duration-only 69.7%
- answer mix: `it was assisted by a teammate` 69.7%; `it was unassisted` 30.3%
- **Label source:** DIRECT — the slot's boolean `assisted` field (BARD sets it only on made field goals).
- Only defined for made field goals. No duration leak -- the duration rule collapses onto the majority class exactly.

```
video: nop-vs-uta-0022400872/165.mp4
Q: The field goal by the player wearing the purple number 34 jersey goes in. Was it assisted by a teammate?
   (A) it was unassisted
   (B) it was assisted by a teammate
A: (B) it was assisted by a teammate
```

### `rebound_side`

- **evaluation_type** `mcq` · **fine_category** Rebounding
- **3,599 rows** · 2 options · chance 50.0% · majority 87.0% · duration-only 86.4%
- answer mix: `the defending team gets the defensive rebound` 87.0%; `the shooting team gets the offensive rebound` 13.0%
- **Label source:** DERIVED — no BARD field names the rebounding team; offensive iff the rebounder's `color` equals the shooter's `color`. Trustable because both colours are BARD fields on the same clip, every ambiguous case is excluded (next bullet), and the one colour-corrupted game is removed by the build-time gate.
- Excludes 738 clips with 0 or >1 shots, 345 multi-rebound clips and 27 where the rebound precedes the shot. IMBALANCED 87.0% defensive, so macro accuracy is the only interpretable headline. No duration leak.

```
video: phi-vs-uta-0022400433/384.mp4
Q: A shot is missed and rebounded in this video. Which team gets the rebound?
   (A) the shooting team gets the offensive rebound
   (B) the defending team gets the defensive rebound
A: (B) the defending team gets the defensive rebound
```

### `block_present`

- **evaluation_type** `mcq` · **fine_category** Shot Outcome
- **4,249 rows** · 2 options · chance 50.0% · majority 90.8% · duration-only 90.5%
- answer mix: `the shot is not blocked` 90.8%; `the shot is blocked` 9.2%
- **Label source:** DERIVED — a presence test: blocked iff any slot in the clip has `action == Block`. The 'not blocked' side is negative evidence; trustable because Block is a first-class action in the same list, so no Block slot means no block was credited in the play-by-play.
- Degenerate: a model that always answers 'not blocked' scores 90.8%; report macro accuracy.

```
video: nop-vs-uta-0022400872/37.mp4
Q: The field goal attempt in this video is missed. Is the shot blocked by a defender?
   (A) the shot is not blocked
   (B) the shot is blocked
A: (A) the shot is not blocked
```

## Match Structure & Reasoning

### `possession_loss`

- **evaluation_type** `mcq` · **fine_category** Possession Outcome
- **2,267 rows** · 3 options · chance 33.3% · majority 63.6% · duration-only 63.1%
- answer mix: `a foul is committed` 63.6%; `a defender steals the ball` 19.2%; `a turnover that is not forced by a steal` 17.2%
- **Label source:** DERIVED — set membership over the clip's `action` fields: {Foul}, {Turnover} and {Steal, Turnover} map one-to-one onto the three options. 'Not forced by a steal' is negative evidence; trustable because steals are logged in the same list as the turnover they force, and every mixed set is excluded.
- No-shot clips whose action set is exactly {Foul}, {Turnover} or {Steal, Turnover}; 364 mixed no-shot clips excluded. No duration leak.

```
video: gsw-vs-nyk-0022400885/35.mp4
Q: No shot is attempted in this video. Which of the following happens?
   (A) a turnover that is not forced by a steal
   (B) a defender steals the ball
   (C) a foul is committed
A: (C) a foul is committed
```

### `action_order`

- **evaluation_type** `mcq` · **fine_category** Play Structure
- **6,127 rows** · 2–5 options (median 2) · chance 46.0% · majority 39.3% · duration-only 40.5%
- answer mix: `a two point shot` 39.3%; `a three point shot` 37.4%; `a turnover` 14.0%; `a free throw` 4.1% …
- **Label source:** DERIVED — the `action` field of the first (or last) slot, using list order as temporal order. Trustable because the list is play-by-play order: in the 3,626 clips with exactly one shot and one rebound, the rebound slot follows the shot slot 99.3% of the time, and the 27 violations are known outliers.
- Clips holding at least two DISTINCT action types. Options are the action types actually present, so the model must order them rather than detect them.
- The LAST variant (`action_order_last`) is much weaker: the last action is a rebound in 59.6% of clips and duration-only reaches 59.8%, so report it separately from FIRST.

```
video: por-vs-phx-0022401219/346.mp4
Q: Which of these actions happens FIRST in this video?
   (A) a two point shot
   (B) a rebound
   (C) a three point shot
A: (C) a three point shot
```

## Player Identification

### `actor_jersey_number`

- **evaluation_type** `mcq` · **fine_category** Player Identification
- **21,416 rows** · 4 options · chance 25.0% · majority **n/a** · duration-only **n/a**
- 55 distinct answers (per-instance, not a closed vocabulary), so there is no majority class to collapse onto — options are shuffled, so chance is the floor
- **Label source:** DIRECT — the slot's `player` field (the colour in the question is the same slot's `color`). Distractors cannot be a second correct answer because the clip has exactly one actor for the asked action type.
- Distractors are other jersey numbers worn by the SAME team in the same game (7-15 available per team per game), so the colour cannot be used to eliminate options.

```
video: chi-vs-tor-0022401223/390.mp4
Q: Which red-jersey player commits a foul in this video?
   (A) number 14
   (B) number 9
   (C) number 2
   (D) number 17
A: (B) number 9
```

### `actor_team_colour`

- **evaluation_type** `mcq` · **fine_category** Team Attribution
- **21,416 rows** · 2 options · chance 50.0% · majority 40.3% · duration-only 39.3%
- answer mix: `white` 40.3%; `blue` 20.5%; `black` 15.0%; `red` 11.1% …
- **Label source:** DIRECT — the slot's `color` field; the two options are the game's two observed colours, aggregated from the same field across the game's clips.
- Options are the two jersey colours actually worn in that game (8-colour vocabulary overall). Tests whether the model localises the acting player rather than just classifying the event.

```
video: hou-vs-okc-0022400879/356.mp4
Q: What colour jersey is worn by the player who commits a turnover in this video?
   (A) white
   (B) black
A: (A) white
```

### `assister_jersey_number`

- **evaluation_type** `mcq` · **fine_category** Play Construction
- **3,198 rows** · 4 options · chance 25.0% · majority **n/a** · duration-only **n/a**
- 52 distinct answers (per-instance, not a closed vocabulary), so there is no majority class to collapse onto — options are shuffled, so chance is the floor
- **Label source:** DIRECT — the slot's `other_player` field (set iff `assisted` is true). Distractors are DERIVED same-colour numbers as in `actor_jersey_number`; unambiguous because no clip has two assisted baskets by the same (colour, number) — verified, zero cases.
- Only the assisted baskets: 3,198 of 3,200 (2 rows in min-vs-sas-0022401218 carry the corrupt assister jersey `'ko'` and are unbuildable). The assister usually performs no other logged action, so the number must be read off the passer rather than recalled from the action list.

```
video: min-vs-sas-0022401218/13.mp4
Q: The basket scored by the player wearing the green number 27 jersey was assisted. Which teammate made the assist?
   (A) number 30
   (B) number 4
   (C) number 10
   (D) number 3
A: (C) number 10
```

## Commentary & Reporting

### `clip_caption`

- **evaluation_type** `open_ended` · **fine_category** Structured Reporting
- **14,676 rows** · free-text target, scored by the LLM judge
- **Label source:** DIRECT — verbatim `captions/caption.json`, BARD's own ground truth; caption and the CSV action fields are provably the same information (next bullet).
- Verified a deterministic render of the action list on all 14,676 clips; keeps comparability with published BARD numbers.

```
video: mem-vs-nop-0022400923/208.mp4
Q: Analyze the basketball video and extract all player actions. For each action, identify the player's jersey number, jersey colour, and action type, using the format: "A player with jersey number <n> and jersey_color <colour> made a <action> which result was made/miss and was assisted by other player with jersey number <n>".
A: A player with jersey number 5 and jersey_color white made a 2PT Shot which result was made and was assisted by other player with jersey number 22
A player with jersey number 13 and jersey_color blue made a Violation
```

### `player_action_description`

- **evaluation_type** `open_ended` · **fine_category** Action Attribution
- **20,720 rows** · free-text target, scored by the LLM judge
- **Label source:** DIRECT fields, templated — a deterministic sentence over the addressed slot's `color`, `player`, `action` and `result`; no inference beyond wording (same uniqueness guarantee as `player_action`).
- Short, targeted, single-fact answers, which an LLM judge scores far more reliably than a multi-clause caption.

```
video: sas-vs-phi-0022400394/213.mp4
Q: Describe what the player wearing the black number 3 jersey does in this video.
A: The player wearing the black number 3 jersey commits a foul.
```

## Caveats to carry into the results table

- Duration leak: misses run longer because BARD ends a missed-shot clip after the rebound. On field goals alone
  a length-only rule scores 62.1% (clip-level) and 58.8% (targeted); free throws mask this in the table above.
  Always report the duration-only baseline next to model scores.
- `rebound_side` and `block_present` are heavily imbalanced: report macro accuracy and a confusion matrix.
- Counts above are pre-gate corpus totals. The build excludes `nyk-vs-mia-0022400869` (both teams intermittently
  labelled `white`) from the colour- and identity-dependent classes; post-gate rows:
  `player_action`/`player_action_description` 20,401, `shot_type_targeted` and `shot_result_targeted` 12,464,
  `shot_assisted` 4,428, `rebound_side` 3,538, `actor_jersey_number`/`actor_team_colour` 21,079,
  `assister_jersey_number` 3,142. Clip-level classes and `clip_caption` keep the game.
- Commentary audio states shot results, shot types and player names; report audio-on and audio-off separately.
