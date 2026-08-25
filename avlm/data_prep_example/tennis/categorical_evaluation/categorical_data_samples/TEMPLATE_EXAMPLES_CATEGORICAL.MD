# Categorical Template Examples

This generated document contains one question-and-answer example per categorical template.

- Example corpus: `training_tennis_data_output/categorical_mcq_qa_sample` (**4** video files; **21,258** generated records)
- Fallback preview corpus: `raw_tennis_data_input` (**2** video files; **20** points)
- Statistics corpus: `aggregated_data_complete` (**252** video files; **50,732** points)
- Standalone config: `configs/generation/tennis_categorical_eval_config.json`
- Templates: **39**
- Evaluation class families: **34**
- Count note: The four sampled none-correct variants share their base class families, and the two rules-knowledge slots share one HF class.
- Requested examples per template: **1**

## Statistics conventions

- Corpus: `aggregated_data_complete` (**252 video files; 50,732 annotated points**).
- Per-template statistics below use the full statistics corpus (`aggregated_data_complete`).
- Example blocks use the latest generated categorical sample where available; rules-knowledge examples use the annotation-triggered question bank.
- Eligible points are points with the required source annotations and a resolvable correct answer.
- Expected data points equal eligible points after applying a template's configured generation probability. They are estimates for probabilistically sampled templates; an actual run can differ slightly.
- Counts are annotation-derived capacity estimates.
- Source-field distributions are computed among eligible points only and show at most the 10 most frequent values; remaining observations are reported as `other values`.
- Every template also reports the correct/composed answer distribution (e.g. Yes/No for audio presence, or the joint label for multi-field templates).
- For `audio_cues`, free-text lists are treated as a set: statistics are per individual cue (point-presence rates), so percentages can sum to more than 100%.

# Fine-Grained Visual Recognition

## `winner_shot_type`

- Fine category: **Shot Mechanics**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`fixed`**
- Implementation type: **`categorical`**
- Example source: latest generated categorical sample
- Source fields: `winner_shot_type`
- Question template: `What type of shot did the winning player hit to win the point? (<player1> - <player1_description>, <player2> - <player2_description>)`
- Option vocabulary source: Prescreen winner_shot_type only; resolve other from winner_shot_type_other; lowercase and canonicalize aliases; exclude unavailable values; and retain distractor values observed at least 5 times. The raw resolved answer remains valid even below this threshold.

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_WinnerShotType_ead80da6-fe16-4008-967c-365f11608259`
- Question: What type of shot did the winning player hit to win the point? (Jil Teichmann - "Jil Teichmann" is the player wearing a white cap, a black top, a black skirt, and white and tan shoes; Victoria Mboko - "Victoria Mboko" is the player wearing a white cap, white top, a green skirt, and white shoes.)
  - (A) drop shot
  - (B) flat groundstroke
  - (C) slice
  - (D) groundstroke with heavy topspin
- Answer: (B) flat groundstroke

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **14,571** (28.72%)
- Expected generated data points: **14,571** (generation probability: 1.00)

**Source field `winner_shot_type` distribution among eligible points** (42 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `flat groundstroke` | 5,833 | 40.03% |
| `ace down the middle` | 1,789 | 12.28% |
| `volley` | 1,590 | 10.91% |
| `groundstroke with heavy topspin` | 1,288 | 8.84% |
| `wide ace` | 1,249 | 8.57% |
| `overhead smash` | 763 | 5.24% |
| `passing shot` | 652 | 4.47% |
| `ace` | 502 | 3.45% |
| `drop shot` | 448 | 3.07% |
| `slice` | 179 | 1.23% |
| _Other values (32 unique)_ | 278 | 1.91% |

## `loser_shot_type`

- Fine category: **Shot Mechanics**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`fixed`**
- Implementation type: **`categorical`**
- Example source: latest generated categorical sample
- Source fields: `loser_shot_type`
- Question template: `What type of shot did the losing player hit at the end of the point? (<player1> - <player1_description>, <player2> - <player2_description>)`
- Option vocabulary source: Prescreen loser_shot_type only; resolve other from loser_shot_type_other; lowercase and canonicalize aliases; exclude unavailable values; and retain distractor values observed at least 5 times. The raw resolved answer remains valid even below this threshold.

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_LoserShotType_3d8f80d3-77fd-4d7a-ba97-9187f476d0fe`
- Question: What type of shot did the losing player hit at the end of the point? (Jil Teichmann - "Jil Teichmann" is the player wearing a white cap, a black top, a black skirt, and white and tan shoes; Victoria Mboko - "Victoria Mboko" is the player wearing a white cap, white top, a green skirt, and white shoes.)
  - (A) overhead serve
  - (B) mishit shot
  - (C) lob
  - (D) passing shot
- Answer: (C) lob

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **29,539** (58.23%)
- Expected generated data points: **29,539** (generation probability: 1.00)

**Source field `loser_shot_type` distribution among eligible points** (71 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `flat groundstroke` | 18,789 | 63.61% |
| `groundstroke with heavy topspin` | 3,252 | 11.01% |
| `lob` | 2,042 | 6.91% |
| `slice` | 968 | 3.28% |
| `block shot` | 805 | 2.73% |
| `volley` | 781 | 2.64% |
| `wide serve` | 692 | 2.34% |
| `half-volley` | 667 | 2.26% |
| `serve down the middle` | 525 | 1.78% |
| `kick serve` | 207 | 0.70% |
| _Other values (61 unique)_ | 811 | 2.75% |

## `winner_shot_type_none`

- Fine category: **Shot Mechanics**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`fixed`**
- Implementation type: **`categorical`**
- Example source: latest generated categorical sample
- Source fields: `winner_shot_type`
- Question template: `What type of shot did the winning player hit to win the point? (<player1> - <player1_description>, <player2> - <player2_description>)`
- Option vocabulary source: Sampled none-correct variant. Prescreen winner_shot_type only; resolve other from winner_shot_type_other; lowercase and canonicalize aliases; exclude unavailable values; and retain distractors observed at least 5 times. The raw resolved answer is intentionally omitted from the choices.

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_WinnerShotTypeNone_ead80da6-fe16-4008-967c-365f11608259`
- Question: What type of shot did the winning player hit to win the point? (Jil Teichmann - "Jil Teichmann" is the player wearing a white cap, a black top, a black skirt, and white and tan shoes; Victoria Mboko - "Victoria Mboko" is the player wearing a white cap, white top, a green skirt, and white shoes.)
  - (A) wide ace
  - (B) half-volley
  - (C) none of the other options
  - (D) passing shot
- Answer: (C) none of the other options

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **14,571** (28.72%)
- Expected generated data points: **2,186** (generation probability: 0.15)

**Correct-answer / composed-answer distribution among eligible points** (1 unique answers)

| Answer | Count | Percent |
|---|---:|---:|
| `none of the other options` | 14,571 | 100.00% |

**Source field `winner_shot_type` distribution among eligible points** (42 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `flat groundstroke` | 5,833 | 40.03% |
| `ace down the middle` | 1,789 | 12.28% |
| `volley` | 1,590 | 10.91% |
| `groundstroke with heavy topspin` | 1,288 | 8.84% |
| `wide ace` | 1,249 | 8.57% |
| `overhead smash` | 763 | 5.24% |
| `passing shot` | 652 | 4.47% |
| `ace` | 502 | 3.45% |
| `drop shot` | 448 | 3.07% |
| `slice` | 179 | 1.23% |
| _Other values (32 unique)_ | 278 | 1.91% |

## `loser_shot_type_none`

- Fine category: **Shot Mechanics**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`fixed`**
- Implementation type: **`categorical`**
- Example source: latest generated categorical sample
- Source fields: `loser_shot_type`
- Question template: `What type of shot did the losing player hit at the end of the point? (<player1> - <player1_description>, <player2> - <player2_description>)`
- Option vocabulary source: Sampled none-correct variant. Prescreen loser_shot_type only; resolve other from loser_shot_type_other; lowercase and canonicalize aliases; exclude unavailable values; and retain distractors observed at least 5 times. The raw resolved answer is intentionally omitted from the choices.

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_LoserShotTypeNone_aab1aa52-d81e-4360-9bd2-fb7c29ab823d`
- Question: What type of shot did the losing player hit at the end of the point? (Jil Teichmann - "Jil Teichmann" is the player wearing a white cap, a black top, a black skirt, and white and tan shoes; Victoria Mboko - "Victoria Mboko" is the player wearing a white cap, white top, a green skirt, and white shoes.)
  - (A) none of the other options
  - (B) kick serve
  - (C) overhead smash
  - (D) groundstroke with heavy topspin
- Answer: (A) none of the other options

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **29,539** (58.23%)
- Expected generated data points: **4,431** (generation probability: 0.15)

**Correct-answer / composed-answer distribution among eligible points** (1 unique answers)

| Answer | Count | Percent |
|---|---:|---:|
| `none of the other options` | 29,539 | 100.00% |

**Source field `loser_shot_type` distribution among eligible points** (71 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `flat groundstroke` | 18,789 | 63.61% |
| `groundstroke with heavy topspin` | 3,252 | 11.01% |
| `lob` | 2,042 | 6.91% |
| `slice` | 968 | 3.28% |
| `block shot` | 805 | 2.73% |
| `volley` | 781 | 2.64% |
| `wide serve` | 692 | 2.34% |
| `half-volley` | 667 | 2.26% |
| `serve down the middle` | 525 | 1.78% |
| `kick serve` | 207 | 0.70% |
| _Other values (61 unique)_ | 811 | 2.75% |

## `winner_hand_used`

- Fine category: **Shot Mechanics**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`fixed`**
- Implementation type: **`categorical`**
- Example source: latest generated categorical sample
- Source fields: `winner_hand_used`
- Question template: `What stroke side did the winning player use on the point-winning shot? (<player1> - <player1_description>, <player2> - <player2_description>)`

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_WinnerHandUsed_ead80da6-fe16-4008-967c-365f11608259`
- Question: What stroke side did the winning player use on the point-winning shot? (Jil Teichmann - "Jil Teichmann" is the player wearing a white cap, a black top, a black skirt, and white and tan shoes; Victoria Mboko - "Victoria Mboko" is the player wearing a white cap, white top, a green skirt, and white shoes.)
  - (A) forehand
  - (B) backhand
- Answer: (A) forehand

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **14,552** (28.68%)
- Expected generated data points: **14,552** (generation probability: 1.00)

**Source field `winner_hand_used` distribution among eligible points** (11 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `forehand` | 11,056 | 75.98% |
| `backhand` | 3,479 | 23.91% |
| _Other values (9 unique)_ | 17 | 0.12% |

## `loser_hand_used`

- Fine category: **Shot Mechanics**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`fixed`**
- Implementation type: **`categorical`**
- Example source: latest generated categorical sample
- Source fields: `loser_hand_used`
- Question template: `What stroke side did the losing player use on the final shot? (<player1> - <player1_description>, <player2> - <player2_description>)`

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_LoserHandUsed_3d8f80d3-77fd-4d7a-ba97-9187f476d0fe`
- Question: What stroke side did the losing player use on the final shot? (Jil Teichmann - "Jil Teichmann" is the player wearing a white cap, a black top, a black skirt, and white and tan shoes; Victoria Mboko - "Victoria Mboko" is the player wearing a white cap, white top, a green skirt, and white shoes.)
  - (A) backhand
  - (B) forehand
- Answer: (B) forehand

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **29,548** (58.24%)
- Expected generated data points: **29,548** (generation probability: 1.00)

**Source field `loser_hand_used` distribution among eligible points** (12 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `forehand` | 16,552 | 56.02% |
| `backhand` | 12,980 | 43.93% |
| _Other values (10 unique)_ | 16 | 0.05% |

## `winner_shot_trajectory`

- Fine category: **Shot Mechanics**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`fixed`**
- Implementation type: **`categorical`**
- Example source: latest generated categorical sample
- Source fields: `winner_shot_trajectory`
- Question template: `What was the winning shot's direction? (<player1> - <player1_description>, <player2> - <player2_description>)`
- Option vocabulary source: Prescreen winner_shot_trajectory only; resolve other from winner_shot_trajectory_other; lowercase and canonicalize aliases; exclude unavailable values; and retain distractor values observed at least 5 times. The raw resolved answer remains valid even below this threshold.

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_WinnerShotTrajectory_ead80da6-fe16-4008-967c-365f11608259`
- Question: What was the winning shot's direction? (Jil Teichmann - "Jil Teichmann" is the player wearing a white cap, a black top, a black skirt, and white and tan shoes; Victoria Mboko - "Victoria Mboko" is the player wearing a white cap, white top, a green skirt, and white shoes.)
  - (A) cross-court
  - (B) no man's right
  - (C) none of the other options
  - (D) ad court
- Answer: (A) cross-court

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **14,554** (28.69%)
- Expected generated data points: **14,554** (generation probability: 1.00)

**Source field `winner_shot_trajectory` distribution among eligible points** (23 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `cross-court` | 6,373 | 43.79% |
| `down the line` | 5,495 | 37.76% |
| `down the middle` | 2,267 | 15.58% |
| `deuce court` | 124 | 0.85% |
| `ad court` | 112 | 0.77% |
| `no man's left` | 80 | 0.55% |
| `no man's right` | 79 | 0.54% |
| _Other values (16 unique)_ | 24 | 0.16% |

## `loser_shot_trajectory`

- Fine category: **Shot Mechanics**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`fixed`**
- Implementation type: **`categorical`**
- Example source: latest generated categorical sample
- Source fields: `loser_shot_trajectory`
- Question template: `What was the losing player's final-shot direction? (<player1> - <player1_description>, <player2> - <player2_description>)`
- Option vocabulary source: Prescreen loser_shot_trajectory only; resolve other from loser_shot_trajectory_other; lowercase and canonicalize aliases; exclude unavailable values; and retain distractor values observed at least 5 times. The raw resolved answer remains valid even below this threshold.

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_LoserShotTrajectory_3d8f80d3-77fd-4d7a-ba97-9187f476d0fe`
- Question: What was the losing player's final-shot direction? (Jil Teichmann - "Jil Teichmann" is the player wearing a white cap, a black top, a black skirt, and white and tan shoes; Victoria Mboko - "Victoria Mboko" is the player wearing a white cap, white top, a green skirt, and white shoes.)
  - (A) cross-court
  - (B) out of court
  - (C) down the line
  - (D) none of the other options
- Answer: (A) cross-court

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **28,514** (56.21%)
- Expected generated data points: **28,514** (generation probability: 1.00)

**Source field `loser_shot_trajectory` distribution among eligible points** (93 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `cross-court` | 11,353 | 39.82% |
| `down the line` | 9,536 | 33.44% |
| `down the middle` | 5,568 | 19.53% |
| `ad court` | 786 | 2.76% |
| `deuce court` | 763 | 2.68% |
| `out of court` | 155 | 0.54% |
| `no man's left` | 122 | 0.43% |
| `no man's right` | 107 | 0.38% |
| _Other values (85 unique)_ | 124 | 0.43% |

## `winner_shot_trajectory_none`

- Fine category: **Shot Mechanics**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`fixed`**
- Implementation type: **`categorical`**
- Example source: latest generated categorical sample
- Source fields: `winner_shot_trajectory`
- Question template: `What was the winning shot's direction? (<player1> - <player1_description>, <player2> - <player2_description>)`
- Option vocabulary source: Sampled none-correct variant. Prescreen winner_shot_trajectory only; resolve other from winner_shot_trajectory_other; lowercase and canonicalize aliases; exclude unavailable values; and retain distractors observed at least 5 times. The raw resolved answer is intentionally omitted from the choices.

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_WinnerShotTrajectoryNone_c34fad05-5b3d-494c-baa0-12f87ffcabc5`
- Question: What was the winning shot's direction? (Jil Teichmann - "Jil Teichmann" is the player wearing a white cap, a black top, a black skirt, and white and tan shoes; Victoria Mboko - "Victoria Mboko" is the player wearing a white cap, white top, a green skirt, and white shoes.)
  - (A) down the middle
  - (B) deuce court
  - (C) no man's left
  - (D) none of the other options
- Answer: (D) none of the other options

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **14,554** (28.69%)
- Expected generated data points: **2,183** (generation probability: 0.15)

**Correct-answer / composed-answer distribution among eligible points** (1 unique answers)

| Answer | Count | Percent |
|---|---:|---:|
| `none of the other options` | 14,554 | 100.00% |

**Source field `winner_shot_trajectory` distribution among eligible points** (23 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `cross-court` | 6,373 | 43.79% |
| `down the line` | 5,495 | 37.76% |
| `down the middle` | 2,267 | 15.58% |
| `deuce court` | 124 | 0.85% |
| `ad court` | 112 | 0.77% |
| `no man's left` | 80 | 0.55% |
| `no man's right` | 79 | 0.54% |
| _Other values (16 unique)_ | 24 | 0.16% |

## `loser_shot_trajectory_none`

- Fine category: **Shot Mechanics**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`fixed`**
- Implementation type: **`categorical`**
- Example source: latest generated categorical sample
- Source fields: `loser_shot_trajectory`
- Question template: `What was the losing player's final-shot direction? (<player1> - <player1_description>, <player2> - <player2_description>)`
- Option vocabulary source: Sampled none-correct variant. Prescreen loser_shot_trajectory only; resolve other from loser_shot_trajectory_other; lowercase and canonicalize aliases; exclude unavailable values; and retain distractors observed at least 5 times. The raw resolved answer is intentionally omitted from the choices.

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_LoserShotTrajectoryNone_29034802-c665-425d-ba23-80092d7a5839`
- Question: What was the losing player's final-shot direction? (Jil Teichmann - "Jil Teichmann" is the player wearing a white cap, a black top, a black skirt, and white and tan shoes; Victoria Mboko - "Victoria Mboko" is the player wearing a white cap, white top, a green skirt, and white shoes.)
  - (A) down the line
  - (B) ad court
  - (C) none of the other options
  - (D) no man's left
- Answer: (C) none of the other options

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **28,514** (56.21%)
- Expected generated data points: **4,277** (generation probability: 0.15)

**Correct-answer / composed-answer distribution among eligible points** (1 unique answers)

| Answer | Count | Percent |
|---|---:|---:|
| `none of the other options` | 28,514 | 100.00% |

**Source field `loser_shot_trajectory` distribution among eligible points** (93 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `cross-court` | 11,353 | 39.82% |
| `down the line` | 9,536 | 33.44% |
| `down the middle` | 5,568 | 19.53% |
| `ad court` | 786 | 2.76% |
| `deuce court` | 763 | 2.68% |
| `out of court` | 155 | 0.54% |
| `no man's left` | 122 | 0.43% |
| `no man's right` | 107 | 0.38% |
| _Other values (85 unique)_ | 124 | 0.43% |

## `server_location`

- Fine category: **Court Positioning**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`fixed`**
- Implementation type: **`categorical`**
- Example source: latest generated categorical sample
- Source fields: `server_location`
- Question template: `Was the server on the near or far side of the visible court? (<player1> - <player1_description>, <player2> - <player2_description>)`
- Note: Both near-court and far-court values are well represented in the complete aggregated corpus.

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_ServerLocation_3d8f80d3-77fd-4d7a-ba97-9187f476d0fe`
- Question: Was the server on the near or far side of the visible court? (Jil Teichmann - "Jil Teichmann" is the player wearing a white cap, a black top, a black skirt, and white and tan shoes; Victoria Mboko - "Victoria Mboko" is the player wearing a white cap, white top, a green skirt, and white shoes.)
  - (A) near-court
  - (B) far-court
- Answer: (B) far-court

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **43,871** (86.48%)
- Expected generated data points: **43,871** (generation probability: 1.00)

**Source field `server_location` distribution among eligible points** (2 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `near-court` | 23,478 | 53.52% |
| `far-court` | 20,393 | 46.48% |

## `receiver_location`

- Fine category: **Court Positioning**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`fixed`**
- Implementation type: **`categorical`**
- Example source: latest generated categorical sample
- Source fields: `receiver_location`
- Question template: `Was the receiver on the near or far side of the visible court? (<player1> - <player1_description>, <player2> - <player2_description>)`
- Note: Both near-court and far-court values are well represented in the complete aggregated corpus.

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_ReceiverLocation_3d8f80d3-77fd-4d7a-ba97-9187f476d0fe`
- Question: Was the receiver on the near or far side of the visible court? (Jil Teichmann - "Jil Teichmann" is the player wearing a white cap, a black top, a black skirt, and white and tan shoes; Victoria Mboko - "Victoria Mboko" is the player wearing a white cap, white top, a green skirt, and white shoes.)
  - (A) far-court
  - (B) near-court
- Answer: (B) near-court

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **43,865** (86.46%)
- Expected generated data points: **43,865** (generation probability: 1.00)

**Source field `receiver_location` distribution among eligible points** (2 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `far-court` | 22,670 | 51.68% |
| `near-court` | 21,195 | 48.32% |

## `server_receiver_location`

- Fine category: **Court Positioning**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`derived`**
- Implementation type: **`derived_categorical`**
- Example source: latest generated categorical sample
- Source fields: `server_location, receiver_location`
- Question template: `Which option correctly identifies whether the server and receiver are on the near or far side of the court? (<player1> - <player1_description>, <player2> - <player2_description>)`
- Note: Both server/receiver near-far assignments are represented in the complete aggregated corpus.

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_ServerReceiverLocation_ead80da6-fe16-4008-967c-365f11608259`
- Question: Which option correctly identifies whether the server and receiver are on the near or far side of the court? (Jil Teichmann - "Jil Teichmann" is the player wearing a white cap, a black top, a black skirt, and white and tan shoes; Victoria Mboko - "Victoria Mboko" is the player wearing a white cap, white top, a green skirt, and white shoes.)
  - (A) server near-court; receiver far-court
  - (B) both near-court
  - (C) both far-court
  - (D) server far-court; receiver near-court
- Answer: (D) server far-court; receiver near-court

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **43,864** (86.46%)
- Expected generated data points: **43,864** (generation probability: 1.00)

**Correct-answer / composed-answer distribution among eligible points** (4 unique answers)

| Answer | Count | Percent |
|---|---:|---:|
| `server near-court; receiver far-court` | 22,662 | 51.66% |
| `server far-court; receiver near-court` | 20,384 | 46.47% |
| `server near-court; receiver near-court` | 810 | 1.85% |
| _Other answers (1 unique)_ | 8 | 0.02% |

**Source field `server_location` distribution among eligible points** (2 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `Near-court` | 23,472 | 53.51% |
| `Far-court` | 20,392 | 46.49% |

**Source field `receiver_location` distribution among eligible points** (2 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `Far-court` | 22,670 | 51.68% |
| `Near-court` | 21,194 | 48.32% |

## `loser_lateral_movement`

- Fine category: **Player Movement**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`fixed`**
- Implementation type: **`categorical`**
- Example source: latest generated categorical sample
- Source fields: `loser_lateral_movement`
- Question template: `What was the losing player's lateral movement? (<player1> - <player1_description>, <player2> - <player2_description>)`

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_LoserLateralMovement_3d8f80d3-77fd-4d7a-ba97-9187f476d0fe`
- Question: What was the losing player's lateral movement? (Jil Teichmann - "Jil Teichmann" is the player wearing a white cap, a black top, a black skirt, and white and tan shoes; Victoria Mboko - "Victoria Mboko" is the player wearing a white cap, white top, a green skirt, and white shoes.)
  - (A) no lateral movement
  - (B) left to right
  - (C) right to left
- Answer: (B) left to right

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **43,745** (86.23%)
- Expected generated data points: **43,745** (generation probability: 1.00)

**Source field `loser_lateral_movement` distribution among eligible points** (3 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `right to left` | 16,282 | 37.22% |
| `left to right` | 16,225 | 37.09% |
| `no lateral movement` | 11,238 | 25.69% |

## `loser_depth_movement`

- Fine category: **Player Movement**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`fixed`**
- Implementation type: **`categorical`**
- Example source: latest generated categorical sample
- Source fields: `loser_depth_movement`
- Question template: `How did the losing player move forward or backward? (<player1> - <player1_description>, <player2> - <player2_description>)`

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_LoserDepthMovement_ead80da6-fe16-4008-967c-365f11608259`
- Question: How did the losing player move forward or backward? (Jil Teichmann - "Jil Teichmann" is the player wearing a white cap, a black top, a black skirt, and white and tan shoes; Victoria Mboko - "Victoria Mboko" is the player wearing a white cap, white top, a green skirt, and white shoes.)
  - (A) no depth movement
  - (B) backward
  - (C) forward
- Answer: (A) no depth movement

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **43,971** (86.67%)
- Expected generated data points: **43,971** (generation probability: 1.00)

**Source field `loser_depth_movement` distribution among eligible points** (3 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `no depth movement` | 35,227 | 80.11% |
| `forward` | 5,415 | 12.31% |
| `backward` | 3,329 | 7.57% |

## `loser_combined_movement`

- Fine category: **Player Movement**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`derived`**
- Implementation type: **`derived_categorical`**
- Example source: latest generated categorical sample
- Source fields: `loser_lateral_movement, loser_depth_movement`
- Question template: `Which description best matches the losing player's movement? (<player1> - <player1_description>, <player2> - <player2_description>)`

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_LoserCombinedMovement_ead80da6-fe16-4008-967c-365f11608259`
- Question: Which description best matches the losing player's movement? (Jil Teichmann - "Jil Teichmann" is the player wearing a white cap, a black top, a black skirt, and white and tan shoes; Victoria Mboko - "Victoria Mboko" is the player wearing a white cap, white top, a green skirt, and white shoes.)
  - (A) right to left; no depth movement
  - (B) left to right; forward
  - (C) left to right; no depth movement
  - (D) no lateral movement; no depth movement
- Answer: (A) right to left; no depth movement

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **43,700** (86.14%)
- Expected generated data points: **43,700** (generation probability: 1.00)

**Correct-answer / composed-answer distribution among eligible points** (9 unique answers)

| Answer | Count | Percent |
|---|---:|---:|
| `right to left; no depth movement` | 12,897 | 29.51% |
| `left to right; no depth movement` | 12,828 | 29.35% |
| `no lateral movement; no depth movement` | 9,277 | 21.23% |
| `right to left; forward` | 2,032 | 4.65% |
| `left to right; forward` | 2,031 | 4.65% |
| `left to right; backward` | 1,345 | 3.08% |
| `right to left; backward` | 1,333 | 3.05% |
| `no lateral movement; forward` | 1,324 | 3.03% |
| `no lateral movement; backward` | 633 | 1.45% |

**Source field `loser_lateral_movement` distribution among eligible points** (3 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `right to left` | 16,262 | 37.21% |
| `left to right` | 16,204 | 37.08% |
| `no lateral movement` | 11,234 | 25.71% |

**Source field `loser_depth_movement` distribution among eligible points** (3 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `no depth movement` | 35,002 | 80.10% |
| `forward` | 5,387 | 12.33% |
| `backward` | 3,311 | 7.58% |

## `loser_point_ended`

- Fine category: **Ball Outcome and Placement**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`fixed`**
- Implementation type: **`categorical`**
- Example source: latest generated categorical sample
- Source fields: `loser_point_ended`
- Question template: `How did the point end for the losing player?`

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_LoserPointEnded_3d8f80d3-77fd-4d7a-ba97-9187f476d0fe`
- Question: How did the point end for the losing player?
  - (A) out
  - (B) not contacted by the losing player
  - (C) landed on the losing player's side of the court
  - (D) into the net
- Answer: (A) out

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **42,262** (83.30%)
- Expected generated data points: **42,262** (generation probability: 1.00)

**Source field `loser_point_ended` distribution among eligible points** (4 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `out` | 17,014 | 40.26% |
| `not contacted by the losing player` | 14,465 | 34.23% |
| `into the net` | 9,799 | 23.19% |
| `landed on the losing player's side of the court` | 984 | 2.33% |

## `mcq_caption_action_mismatch`

- Fine category: **Action Consistency**
- Format: **MCQ**
- Option space: **`open`**
- Distractor generation: **`derived`**
- Implementation type: **`derived_categorical`**
- Example source: latest generated categorical sample
- Source fields: `point_caption, winner_shot_type, winner_hand_used, winner_shot_trajectory, loser_shot_type, loser_hand_used, loser_shot_trajectory, loser_lateral_movement, loser_depth_movement`
- Question template: `What happened in this point? Provide a detailed caption including the action of the players. (<player1> - <player1_description>, <player2> - <player2_description>)`

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_McqCaptionActionMismatch_ead80da6-fe16-4008-967c-365f11608259`
- Question: What happened in this point? Provide a detailed caption including the action of the players. (Jil Teichmann - "Jil Teichmann" is the player wearing a white cap, a black top, a black skirt, and white and tan shoes; Victoria Mboko - "Victoria Mboko" is the player wearing a white cap, white top, a green skirt, and white shoes.)
  - (A) The rally starts with the first serve by J. Teichmann that lands in the deuce service court. V. Mboko returns it with a backhand shot, to which J. Teichmann hits a forehand shot that lands the ball into the ad court of V. Mboko. V. Mboko is unable to return the shot. It is a winner by J. Teichman, giving her a point. The winning player's stroke side was backhand.
  - (B) The rally starts with the first serve by J. Teichmann that lands in the deuce service court. V. Mboko returns it with a backhand shot, to which J. Teichmann hits a forehand shot that lands the ball into the ad court of V. Mboko. V. Mboko is unable to return the shot. It is a winner by J. Teichman, giving her a point. The winning player's stroke side was double-handed backhand.
  - (C) The rally starts with the first serve by J. Teichmann that lands in the deuce service court. V. Mboko returns it with a backhand shot, to which J. Teichmann hits a forehand shot that lands the ball into the ad court of V. Mboko. V. Mboko is unable to return the shot. It is a winner by J. Teichman, giving her a point. The winning player's stroke side was forehand.
- Answer: (C) The rally starts with the first serve by J. Teichmann that lands in the deuce service court. V. Mboko returns it with a backhand shot, to which J. Teichmann hits a forehand shot that lands the ball into the ad court of V. Mboko. V. Mboko is unable to return the shot. It is a winner by J. Teichman, giving her a point. The winning player's stroke side was forehand.

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **44,077** (86.88%)
- Expected generated data points: **44,077** (generation probability: 1.00)

**Correct-answer / composed-answer distribution among eligible points** (43,614 unique answers)

| Answer | Count | Percent |
|---|---:|---:|
| _No eligible answers_ | 0 | 0.00% |
| _Other answers (43,614 unique)_ | 44,077 | 100.00% |

**Source field `point_caption` distribution among eligible points** (43,521 unique values)

| Value | Count | Percent |
|---|---:|---:|
| _No non-missing values_ | 0 | 0.00% |
| _Other values (43,521 unique)_ | 44,077 | 100.00% |

**Source field `winner_shot_type` distribution among eligible points** (42 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `flat groundstroke` | 5,833 | 13.23% |
| `ace down the middle` | 1,789 | 4.06% |
| `volley` | 1,590 | 3.61% |
| `groundstroke with heavy topspin` | 1,288 | 2.92% |
| `wide ace` | 1,249 | 2.83% |
| `overhead smash` | 763 | 1.73% |
| `passing shot` | 652 | 1.48% |
| `ace` | 502 | 1.14% |
| `drop shot` | 448 | 1.02% |
| `slice` | 179 | 0.41% |
| _Other values (32 unique)_ | 278 | 0.63% |
| _Missing/unavailable_ | 29,506 | 66.94% |

**Source field `winner_hand_used` distribution among eligible points** (11 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `forehand` | 11,056 | 25.08% |
| `backhand` | 3,479 | 7.89% |
| _Other values (9 unique)_ | 17 | 0.04% |
| _Missing/unavailable_ | 29,525 | 66.99% |

**Source field `winner_shot_trajectory` distribution among eligible points** (23 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `cross-court` | 6,373 | 14.46% |
| `down the line` | 5,495 | 12.47% |
| `down the middle` | 2,267 | 5.14% |
| `deuce court` | 124 | 0.28% |
| `ad court` | 112 | 0.25% |
| `no man's left` | 80 | 0.18% |
| `no man's right` | 79 | 0.18% |
| _Other values (16 unique)_ | 24 | 0.05% |
| _Missing/unavailable_ | 29,523 | 66.98% |

**Source field `loser_shot_type` distribution among eligible points** (71 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `flat groundstroke` | 18,789 | 42.63% |
| `groundstroke with heavy topspin` | 3,252 | 7.38% |
| `lob` | 2,042 | 4.63% |
| `slice` | 968 | 2.20% |
| `block shot` | 805 | 1.83% |
| `volley` | 781 | 1.77% |
| `wide serve` | 692 | 1.57% |
| `half-volley` | 667 | 1.51% |
| `serve down the middle` | 525 | 1.19% |
| `kick serve` | 207 | 0.47% |
| _Other values (61 unique)_ | 811 | 1.84% |
| _Missing/unavailable_ | 14,538 | 32.98% |

**Source field `loser_hand_used` distribution among eligible points** (12 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `forehand` | 16,552 | 37.55% |
| `backhand` | 12,980 | 29.45% |
| _Other values (10 unique)_ | 16 | 0.04% |
| _Missing/unavailable_ | 14,529 | 32.96% |

**Source field `loser_shot_trajectory` distribution among eligible points** (93 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `cross-court` | 11,353 | 25.76% |
| `down the line` | 9,536 | 21.63% |
| `down the middle` | 5,568 | 12.63% |
| `ad court` | 786 | 1.78% |
| `deuce court` | 763 | 1.73% |
| `out of court` | 155 | 0.35% |
| `no man's left` | 122 | 0.28% |
| `no man's right` | 107 | 0.24% |
| _Other values (85 unique)_ | 124 | 0.28% |
| _Missing/unavailable_ | 15,563 | 35.31% |

**Source field `loser_lateral_movement` distribution among eligible points** (28 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `right to left` | 16,282 | 36.94% |
| `left to right` | 16,225 | 36.81% |
| `no lateral movement` | 11,238 | 25.50% |
| `adjustment steps` | 102 | 0.23% |
| _Other values (24 unique)_ | 158 | 0.36% |
| _Missing/unavailable_ | 72 | 0.16% |

**Source field `loser_depth_movement` distribution among eligible points** (9 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `no depth movement` | 35,227 | 79.92% |
| `forward` | 5,415 | 12.29% |
| `backward` | 3,329 | 7.55% |
| _Other values (6 unique)_ | 7 | 0.02% |
| _Missing/unavailable_ | 99 | 0.22% |

# Match Facts

## `server_winner`

- Fine category: **Point Roles and Outcome**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`fixed`**
- Implementation type: **`categorical`**
- Example source: latest generated categorical sample
- Source fields: `server_winner`
- Question template: `Who served and who won this tennis point? (<player1> - <player1_description>, <player2> - <player2_description>)`

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_ServerWinner_ead80da6-fe16-4008-967c-365f11608259`
- Question: Who served and who won this tennis point? (Jil Teichmann - "Jil Teichmann" is the player wearing a white cap, a black top, a black skirt, and white and tan shoes; Victoria Mboko - "Victoria Mboko" is the player wearing a white cap, white top, a green skirt, and white shoes.)
  - (A) Victoria Mboko served and Jil Teichmann won
  - (B) Jil Teichmann served and Victoria Mboko won
  - (C) Victoria Mboko served and Victoria Mboko won
  - (D) Jil Teichmann served and Jil Teichmann won
- Answer: (D) Jil Teichmann served and Jil Teichmann won

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **50,650** (99.84%)
- Expected generated data points: **50,650** (generation probability: 1.00)

**Correct-answer / composed-answer distribution among eligible points** (689 unique answers)

| Answer | Count | Percent |
|---|---:|---:|
| `Rafael Nadal served and Rafael Nadal won` | 1,595 | 3.15% |
| `Novak Djokovic served and Novak Djokovic won` | 1,506 | 2.97% |
| `Roger Federer served and Roger Federer won` | 1,374 | 2.71% |
| `Jannik Sinner served and Jannik Sinner won` | 1,081 | 2.13% |
| `Serena Williams served and Serena Williams won` | 783 | 1.55% |
| `Djokovic served and Djokovic won` | 748 | 1.48% |
| `Alexander Zverev served and Alexander Zverev won` | 697 | 1.38% |
| `Carlos Alcaraz served and Carlos Alcaraz won` | 694 | 1.37% |
| `Ben Shelton served and Ben Shelton won` | 596 | 1.18% |
| `Stefanos Tsitsipas served and Stefanos Tsitsipas won` | 528 | 1.04% |
| _Other answers (679 unique)_ | 41,048 | 81.04% |

**Source field `winner` distribution among eligible points** (2 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `Server` | 30,730 | 60.67% |
| `Receiver` | 19,920 | 39.33% |

## `receiver_winner`

- Fine category: **Point Roles and Outcome**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`fixed`**
- Implementation type: **`categorical`**
- Example source: latest generated categorical sample
- Source fields: `receiver_winner`
- Question template: `Who received the serve and who won this tennis point? (<player1> - <player1_description>, <player2> - <player2_description>)`

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_ReceiverWinner_d2757a9f-1b8a-458f-80ae-5faa3659b846`
- Question: Who received the serve and who won this tennis point? (Jil Teichmann - "Jil Teichmann" is the player wearing a white cap, a black top, a black skirt, and white and tan shoes; Victoria Mboko - "Victoria Mboko" is the player wearing a white cap, white top, a green skirt, and white shoes.)
  - (A) Victoria Mboko received and Jil Teichmann won
  - (B) Jil Teichmann received and Victoria Mboko won
  - (C) Victoria Mboko received and Victoria Mboko won
  - (D) Jil Teichmann received and Jil Teichmann won
- Answer: (C) Victoria Mboko received and Victoria Mboko won

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **50,650** (99.84%)
- Expected generated data points: **50,650** (generation probability: 1.00)

**Correct-answer / composed-answer distribution among eligible points** (690 unique answers)

| Answer | Count | Percent |
|---|---:|---:|
| `Rafael Nadal received and Rafael Nadal won` | 1,011 | 2.00% |
| `Novak Djokovic received and Novak Djokovic won` | 999 | 1.97% |
| `Roger Federer received and Roger Federer won` | 794 | 1.57% |
| `Jannik Sinner received and Jannik Sinner won` | 741 | 1.46% |
| `Serena Williams received and Serena Williams won` | 532 | 1.05% |
| `Carlos Alcaraz received and Carlos Alcaraz won` | 515 | 1.02% |
| `Djokovic received and Djokovic won` | 481 | 0.95% |
| `Novak Djokovic received and Rafael Nadal won` | 452 | 0.89% |
| `Rafael Nadal received and Novak Djokovic won` | 449 | 0.89% |
| `Alexander Zverev received and Alexander Zverev won` | 441 | 0.87% |
| _Other answers (680 unique)_ | 44,235 | 87.33% |

**Source field `winner` distribution among eligible points** (2 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `Server` | 30,730 | 60.67% |
| `Receiver` | 19,920 | 39.33% |

## `how_point_ended`

- Fine category: **Point Roles and Outcome**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`fixed`**
- Implementation type: **`categorical`**
- Example source: latest generated categorical sample
- Source fields: `how_point_ended`
- Question template: `How did this tennis point end?`

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_PointEnd_ead80da6-fe16-4008-967c-365f11608259`
- Question: How did this tennis point end?
  - (A) Retirement/Default
  - (B) Winner
  - (C) Let
  - (D) Ace
- Answer: (B) Winner

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **50,682** (99.90%)
- Expected generated data points: **50,682** (generation probability: 1.00)

**Correct-answer / composed-answer distribution among eligible points** (8 unique answers)

| Answer | Count | Percent |
|---|---:|---:|
| `Unforced Error` | 26,861 | 53.00% |
| `Winner` | 12,535 | 24.73% |
| `Forced Error` | 5,046 | 9.96% |
| `Ace` | 3,982 | 7.86% |
| `Double Fault` | 1,969 | 3.89% |
| `Point ended by review` | 221 | 0.44% |
| `Violation` | 61 | 0.12% |
| _Other answers (1 unique)_ | 7 | 0.01% |

**Source field `how_point_ended` distribution among eligible points** (9 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `Unforced Error: player makes a mistake during the point without significant pressure such as hitting the ball outside of the court (long or wide) or into the net during a routine shot. Routine miss without pressure.` | 26,861 | 53.00% |
| `Winner: When a shot (other than serve) lands in and isn’t touched or can’t be returned. Clean shot not returned` | 11,441 | 22.57% |
| `Forced Error: error caused due to pressure from the opponent’s aggressive or well-placed shot. e.g., rushed forehand sent long after a powerful serve or wide angle. The opponent pressured into an error.` | 5,046 | 9.96% |
| `Ace: is when the server scores in the first shot. The receiver didn’t touch the ball.` | 3,982 | 7.86% |
| `Double Fault: The server misses both first and second serves, resulting in an automatic point for the opponent.` | 1,969 | 3.89% |
| `Winner: When a shot (other than serve) lands in and isn’t touched or can’t be returned. Clean shot not returned (excluding net winners)` | 1,094 | 2.16% |
| `Point ended by review` | 221 | 0.44% |
| `Violation: Rule-based loss of point includes player touching net, ball touches player, double bounce, foot fault, etc.` | 61 | 0.12% |
| _Other values (1 unique)_ | 7 | 0.01% |

## `winner_role`

- Fine category: **Point Roles and Outcome**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`fixed`**
- Implementation type: **`categorical`**
- Example source: latest generated categorical sample
- Source fields: `winner`
- Question template: `Did the server or receiver win the point?`

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_WinnerRole_d2757a9f-1b8a-458f-80ae-5faa3659b846`
- Question: Did the server or receiver win the point?
  - (A) receiver
  - (B) server
- Answer: (A) receiver

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **50,689** (99.92%)
- Expected generated data points: **50,689** (generation probability: 1.00)

**Source field `winner` distribution among eligible points** (2 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `server` | 30,754 | 60.67% |
| `receiver` | 19,935 | 39.33% |

## `score_after`

- Fine category: **Score and Serve State**
- Format: **MCQ**
- Option space: **`open`**
- Distractor generation: **`heuristic`**
- Implementation type: **`freeform_with_distractors`**
- Example source: latest generated categorical sample
- Source fields: `score_after`
- Question template: `<score_before>. What was the score after this point? (<player1> - <player1_description>, <player2> - <player2_description>)`

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_ScoreAfter_d2757a9f-1b8a-458f-80ae-5faa3659b846`
- Question: J. Teichmann vs. V. Mboko:{0-0, 15-0}. What was the score after this point? (Jil Teichmann - "Jil Teichmann" is the player wearing a white cap, a black top, a black skirt, and white and tan shoes; Victoria Mboko - "Victoria Mboko" is the player wearing a white cap, white top, a green skirt, and white shoes.)
  - (A) J. Teichmann vs. V. Mboko:{0-0, 15-15}
  - (B) J. Teichmann vs. V. Mboko:{0-0, 40-0}
  - (C) J. Teichmann vs. V. Mboko:{0-0, 30-0}
  - (D) J. Teichmann vs. V. Mboko:{0-0, 15-0}
- Answer: (A) J. Teichmann vs. V. Mboko:{0-0, 15-15}

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **50,724** (99.98%)
- Expected generated data points: **50,724** (generation probability: 1.00)

**Correct-answer / composed-answer distribution among eligible points** (47,470 unique answers)

| Answer | Count | Percent |
|---|---:|---:|
| _No eligible answers_ | 0 | 0.00% |
| _Other answers (47,470 unique)_ | 50,724 | 100.00% |

## `num_serving_attempts_until_successful`

- Fine category: **Score and Serve State**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`fixed`**
- Implementation type: **`categorical`**
- Example source: latest generated categorical sample
- Source fields: `num_serving_attempts_until_successful`
- Question template: `How many serving attempts were made in this point?`

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_ServingAttempts_d2757a9f-1b8a-458f-80ae-5faa3659b846`
- Question: How many serving attempts were made in this point?
  - (A) 1
  - (B) 4
  - (C) 2
  - (D) 3
- Answer: (A) 1

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **50,713** (99.96%)
- Expected generated data points: **50,713** (generation probability: 1.00)

**Source field `num_serving_attempts_until_successful` distribution among eligible points** (6 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `1` | 30,455 | 60.05% |
| `2` | 19,323 | 38.10% |
| `3` | 880 | 1.74% |
| `4` | 49 | 0.10% |
| _Other values (2 unique)_ | 6 | 0.01% |

## `num_shots_exchanged`

- Fine category: **Rally Structure**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`fixed`**
- Implementation type: **`numeric_range`**
- Example source: latest generated categorical sample
- Source fields: `num_shots_exchanged`
- Question template: `How many shots were exchanged in this rally?`

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_RallyLength_ead80da6-fe16-4008-967c-365f11608259`
- Question: How many shots were exchanged in this rally?
  - (A) 21+ shots (Extended rally)
  - (B) 8 shots
  - (C) 3 shots
  - (D) 11-20 shots (Long rally)
- Answer: (C) 3 shots

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **50,713** (99.96%)
- Expected generated data points: **50,713** (generation probability: 1.00)

**Correct-answer / composed-answer distribution among eligible points** (13 unique answers)

| Answer | Count | Percent |
|---|---:|---:|
| `3 shots` | 8,532 | 16.82% |
| `2 shots` | 8,425 | 16.61% |
| `4 shots` | 6,204 | 12.23% |
| `5 shots` | 5,154 | 10.16% |
| `11-20 shots (Long rally)` | 4,354 | 8.59% |
| `1 shot` | 4,019 | 7.92% |
| `6 shots` | 3,636 | 7.17% |
| `7 shots` | 2,941 | 5.80% |
| `8 shots` | 2,181 | 4.30% |
| `0 shots` | 1,987 | 3.92% |
| _Other answers (3 unique)_ | 3,280 | 6.47% |

**Source field `num_shots_exchanged` distribution among eligible points** (44 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `3` | 8,532 | 16.82% |
| `2` | 8,425 | 16.61% |
| `4` | 6,204 | 12.23% |
| `5` | 5,154 | 10.16% |
| `1` | 4,019 | 7.92% |
| `6` | 3,636 | 7.17% |
| `7` | 2,941 | 5.80% |
| `8` | 2,181 | 4.30% |
| `0` | 1,987 | 3.92% |
| `9` | 1,607 | 3.17% |
| _Other values (34 unique)_ | 6,027 | 11.88% |

## `rally_length_bucket`

- Fine category: **Rally Structure**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`fixed`**
- Implementation type: **`numeric_range`**
- Example source: latest generated categorical sample
- Source fields: `num_shots_exchanged`
- Question template: `How long was this rally?`

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_RallyLengthBucket_ead80da6-fe16-4008-967c-365f11608259`
- Question: How long was this rally?
  - (A) Medium rally (5–10 shots)
  - (B) Extended rally (21+ shots)
  - (C) Short rally (1–4 shots)
  - (D) Serve-only / no exchange (0 shots)
- Answer: (C) Short rally (1–4 shots)

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **50,713** (99.96%)
- Expected generated data points: **50,713** (generation probability: 1.00)

**Correct-answer / composed-answer distribution among eligible points** (5 unique answers)

| Answer | Count | Percent |
|---|---:|---:|
| `Short rally (1–4 shots)` | 27,180 | 53.60% |
| `Medium rally (5–10 shots)` | 16,796 | 33.12% |
| `Long rally (11–20 shots)` | 4,354 | 8.59% |
| `Serve-only / no exchange (0 shots)` | 1,987 | 3.92% |
| `Extended rally (21+ shots)` | 396 | 0.78% |

**Source field `num_shots_exchanged` distribution among eligible points** (44 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `3` | 8,532 | 16.82% |
| `2` | 8,425 | 16.61% |
| `4` | 6,204 | 12.23% |
| `5` | 5,154 | 10.16% |
| `1` | 4,019 | 7.92% |
| `6` | 3,636 | 7.17% |
| `7` | 2,941 | 5.80% |
| `8` | 2,181 | 4.30% |
| `0` | 1,987 | 3.92% |
| `9` | 1,607 | 3.17% |
| _Other values (34 unique)_ | 6,027 | 11.88% |

## `score_transition`

- Fine category: **Score and Serve State**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`derived`**
- Implementation type: **`derived_categorical`**
- Example source: latest generated categorical sample
- Source fields: `score_before, score_after, winner, server, receiver`
- Question template: `The score before the point was <score_before>. Which option correctly identifies the point winner and the new score?`

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_ScoreTransition_ead80da6-fe16-4008-967c-365f11608259`
- Question: The score before the point was J. Teichmann vs. V. Mboko:{0-0, 0-0}. Which option correctly identifies the point winner and the new score?
  - (A) Victoria Mboko(receiver)won the point; new score: J. Teichmann vs. V. Mboko:{0-0, 15-0}
  - (B) Jil Teichmann(server)won the point; new score: J. Teichmann vs. V. Mboko:{0-0, 15-0}
  - (C) Jil Teichmann(receiver)won the point; new score: J. Teichmann vs. V. Mboko:{0-0, 15-0}
  - (D) Jil Teichmann(server)won the point; new score: J. Teichmann vs. V. Mboko:{0-0, 0-15}
- Answer: (B) Jil Teichmann(server)won the point; new score: J. Teichmann vs. V. Mboko:{0-0, 15-0}

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **50,541** (99.62%)
- Expected generated data points: **50,541** (generation probability: 1.00)

**Correct-answer / composed-answer distribution among eligible points** (48,230 unique answers)

| Answer | Count | Percent |
|---|---:|---:|
| _No eligible answers_ | 0 | 0.00% |
| _Other answers (48,230 unique)_ | 50,541 | 100.00% |

**Source field `winner` distribution among eligible points** (2 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `Server` | 30,673 | 60.69% |
| `Receiver` | 19,868 | 39.31% |

## `serve_sequence_outcome`

- Fine category: **Score and Serve State**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`derived`**
- Implementation type: **`derived_categorical`**
- Example source: latest generated categorical sample
- Source fields: `num_serving_attempts_until_successful, how_point_ended`
- Question template: `Which serve sequence and point outcome correctly describe this point?`

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_ServeSequenceOutcome_ead80da6-fe16-4008-967c-365f11608259`
- Question: Which serve sequence and point outcome correctly describe this point?
  - (A) First and second serves were faults; point outcome: Double Fault
  - (B) First serve was a fault; second serve was an ace; point outcome: Ace
  - (C) First serve was successful; point outcome: Winner
  - (D) First serve was an ace; point outcome: Ace
- Answer: (C) First serve was successful; point outcome: Winner

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **49,423** (97.42%)
- Expected generated data points: **49,423** (generation probability: 1.00)

**Correct-answer / composed-answer distribution among eligible points** (9 unique answers)

| Answer | Count | Percent |
|---|---:|---:|
| `First serve was successful; point outcome: Unforced Error` | 15,108 | 30.57% |
| `First serve was a fault; second serve was successful; point outcome: Unforced Error` | 11,211 | 22.68% |
| `First serve was successful; point outcome: Winner` | 7,970 | 16.13% |
| `First serve was a fault; second serve was successful; point outcome: Winner` | 4,366 | 8.83% |
| `First serve was an ace; point outcome: Ace` | 3,714 | 7.51% |
| `First serve was successful; point outcome: Forced Error` | 3,430 | 6.94% |
| `First and second serves were faults; point outcome: Double Fault` | 1,831 | 3.70% |
| `First serve was a fault; second serve was successful; point outcome: Forced Error` | 1,540 | 3.12% |
| `First serve was a fault; second serve was an ace; point outcome: Ace` | 253 | 0.51% |

**Source field `num_serving_attempts_until_successful` distribution among eligible points** (2 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `1` | 30,222 | 61.15% |
| `2` | 19,201 | 38.85% |

**Source field `how_point_ended` distribution among eligible points** (6 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `Unforced Error: player makes a mistake during the point without significant pressure such as hitting the ball outside of the court (long or wide) or into the net during a routine shot. Routine miss without pressure.` | 26,319 | 53.25% |
| `Winner: When a shot (other than serve) lands in and isn’t touched or can’t be returned. Clean shot not returned` | 11,259 | 22.78% |
| `Forced Error: error caused due to pressure from the opponent’s aggressive or well-placed shot. e.g., rushed forehand sent long after a powerful serve or wide angle. The opponent pressured into an error.` | 4,970 | 10.06% |
| `Ace: is when the server scores in the first shot. The receiver didn’t touch the ball.` | 3,967 | 8.03% |
| `Double Fault: The server misses both first and second serves, resulting in an automatic point for the opponent.` | 1,831 | 3.70% |
| `Winner: When a shot (other than serve) lands in and isn’t touched or can’t be returned. Clean shot not returned (excluding net winners)` | 1,077 | 2.18% |

# Rules & Reasoning

## `serve_rule_consistency`

- Fine category: **Cross-field Reasoning**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`derived`**
- Implementation type: **`derived_categorical`**
- Example source: latest generated categorical sample
- Source fields: `num_serving_attempts_until_successful, how_point_ended`
- Question template: `Which option is consistent with tennis rules and the video?`
- Why reasoning: The model must apply serve rules: a successful second serve cannot be a double fault, and an ace or two faults ends the point immediately. Every distractor violates one of these rules, so this template tests rule reasoning but does not require the video.

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_ServeRuleConsistency_d2757a9f-1b8a-458f-80ae-5faa3659b846`
- Question: Which option is consistent with tennis rules and the video?
  - (A) First serve was successful; point outcome: Unforced Error
  - (B) Both serves were faults, but the rally continued to Unforced Error
  - (C) First serve was a fault, but the rally continued without a second serve
  - (D) First serve was an ace, but the rally continued to Unforced Error
- Answer: (A) First serve was successful; point outcome: Unforced Error

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **49,423** (97.42%)
- Expected generated data points: **49,423** (generation probability: 1.00)

**Correct-answer / composed-answer distribution among eligible points** (9 unique answers)

| Answer | Count | Percent |
|---|---:|---:|
| `First serve was successful; point outcome: Unforced Error` | 15,108 | 30.57% |
| `First serve was a fault; second serve was successful; point outcome: Unforced Error` | 11,211 | 22.68% |
| `First serve was successful; point outcome: Winner` | 7,970 | 16.13% |
| `First serve was a fault; second serve was successful; point outcome: Winner` | 4,366 | 8.83% |
| `First serve was an ace; point outcome: Ace` | 3,714 | 7.51% |
| `First serve was successful; point outcome: Forced Error` | 3,430 | 6.94% |
| `First and second serves were faults; point outcome: Double Fault` | 1,831 | 3.70% |
| `First serve was a fault; second serve was successful; point outcome: Forced Error` | 1,540 | 3.12% |
| `First serve was a fault; second serve was an ace; point outcome: Ace` | 253 | 0.51% |

**Source field `num_serving_attempts_until_successful` distribution among eligible points** (2 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `1` | 30,222 | 61.15% |
| `2` | 19,201 | 38.85% |

**Source field `how_point_ended` distribution among eligible points** (6 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `Unforced Error: player makes a mistake during the point without significant pressure such as hitting the ball outside of the court (long or wide) or into the net during a routine shot. Routine miss without pressure.` | 26,319 | 53.25% |
| `Winner: When a shot (other than serve) lands in and isn’t touched or can’t be returned. Clean shot not returned` | 11,259 | 22.78% |
| `Forced Error: error caused due to pressure from the opponent’s aggressive or well-placed shot. e.g., rushed forehand sent long after a powerful serve or wide angle. The opponent pressured into an error.` | 4,970 | 10.06% |
| `Ace: is when the server scores in the first shot. The receiver didn’t touch the ball.` | 3,967 | 8.03% |
| `Double Fault: The server misses both first and second serves, resulting in an automatic point for the opponent.` | 1,831 | 3.70% |
| `Winner: When a shot (other than serve) lands in and isn’t touched or can’t be returned. Clean shot not returned (excluding net winners)` | 1,077 | 2.18% |

## `grounded_serve_rule_consistency`

- Fine category: **Cross-field Reasoning**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`derived`**
- Implementation type: **`derived_categorical`**
- Example source: latest generated categorical sample
- Source fields: `num_serving_attempts_until_successful, how_point_ended`
- Question template: `Which option is consistent with tennis rules and matches what happened in this point?`
- Why reasoning: The model first applies serve rules to eliminate two contradictory options. Two valid tennis scenarios remain, so it must inspect the video to identify which serve sequence and outcome actually occurred.

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_GroundedServeRuleConsistency_ead80da6-fe16-4008-967c-365f11608259`
- Question: Which option is consistent with tennis rules and matches what happened in this point?
  - (A) First serve was successful; point outcome: Winner
  - (B) First serve was a fault, but the rally continued without a second serve
  - (C) First serve was an ace; point outcome: Ace
  - (D) First serve was an ace, but the rally continued to Winner
- Answer: (A) First serve was successful; point outcome: Winner

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **49,423** (97.42%)
- Expected generated data points: **49,423** (generation probability: 1.00)

**Correct-answer / composed-answer distribution among eligible points** (9 unique answers)

| Answer | Count | Percent |
|---|---:|---:|
| `First serve was successful; point outcome: Unforced Error` | 15,108 | 30.57% |
| `First serve was a fault; second serve was successful; point outcome: Unforced Error` | 11,211 | 22.68% |
| `First serve was successful; point outcome: Winner` | 7,970 | 16.13% |
| `First serve was a fault; second serve was successful; point outcome: Winner` | 4,366 | 8.83% |
| `First serve was an ace; point outcome: Ace` | 3,714 | 7.51% |
| `First serve was successful; point outcome: Forced Error` | 3,430 | 6.94% |
| `First and second serves were faults; point outcome: Double Fault` | 1,831 | 3.70% |
| `First serve was a fault; second serve was successful; point outcome: Forced Error` | 1,540 | 3.12% |
| `First serve was a fault; second serve was an ace; point outcome: Ace` | 253 | 0.51% |

**Source field `num_serving_attempts_until_successful` distribution among eligible points** (2 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `1` | 30,222 | 61.15% |
| `2` | 19,201 | 38.85% |

**Source field `how_point_ended` distribution among eligible points** (6 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `Unforced Error: player makes a mistake during the point without significant pressure such as hitting the ball outside of the court (long or wide) or into the net during a routine shot. Routine miss without pressure.` | 26,319 | 53.25% |
| `Winner: When a shot (other than serve) lands in and isn’t touched or can’t be returned. Clean shot not returned` | 11,259 | 22.78% |
| `Forced Error: error caused due to pressure from the opponent’s aggressive or well-placed shot. e.g., rushed forehand sent long after a powerful serve or wide angle. The opponent pressured into an error.` | 4,970 | 10.06% |
| `Ace: is when the server scores in the first shot. The receiver didn’t touch the ball.` | 3,967 | 8.03% |
| `Double Fault: The server misses both first and second serves, resulting in an automatic point for the opponent.` | 1,831 | 3.70% |
| `Winner: When a shot (other than serve) lands in and isn’t touched or can’t be returned. Clean shot not returned (excluding net winners)` | 1,077 | 2.18% |

## `point_outcome_reasoning`

- Fine category: **Cross-field Reasoning**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`derived`**
- Implementation type: **`derived_categorical`**
- Example source: latest generated categorical sample
- Source fields: `winner, how_point_ended`
- Question template: `Which option correctly explains why the server or receiver won the point?`
- Why reasoning: The model must connect the ending mechanism to the winning role: an error loses the point, a winner or ace wins it, and a double fault awards it to the receiver. Rules remove two causal contradictions; the video distinguishes the two coherent explanations.

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_PointOutcomeReasoning_ead80da6-fe16-4008-967c-365f11608259`
- Question: Which option correctly explains why the server or receiver won the point?
  - (A) receiver hit a point-ending winner and won the point
  - (B) server hit a point-ending winner and won the point
  - (C) server committed the point-ending error but still won the point
  - (D) receiver hit the point-ending winner, but server won the point
- Answer: (B) server hit a point-ending winner and won the point

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **50,351** (99.25%)
- Expected generated data points: **50,351** (generation probability: 1.00)

**Correct-answer / composed-answer distribution among eligible points** (8 unique answers)

| Answer | Count | Percent |
|---|---:|---:|
| `receiver committed an unforced error, so server won the point` | 15,301 | 30.39% |
| `server committed an unforced error, so receiver won the point` | 11,546 | 22.93% |
| `server hit a point-ending winner and won the point` | 7,699 | 15.29% |
| `receiver hit a point-ending winner and won the point` | 4,830 | 9.59% |
| `server hit an ace and won the point immediately` | 3,978 | 7.90% |
| `server forced an error from receiver and won the point` | 3,568 | 7.09% |
| `server double-faulted, so receiver won the point` | 1,956 | 3.88% |
| `receiver forced an error from server and won the point` | 1,473 | 2.93% |

**Source field `winner` distribution among eligible points** (2 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `Server` | 30,546 | 60.67% |
| `Receiver` | 19,805 | 39.33% |

**Source field `how_point_ended` distribution among eligible points** (6 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `Unforced Error: player makes a mistake during the point without significant pressure such as hitting the ball outside of the court (long or wide) or into the net during a routine shot. Routine miss without pressure.` | 26,847 | 53.32% |
| `Winner: When a shot (other than serve) lands in and isn’t touched or can’t be returned. Clean shot not returned` | 11,438 | 22.72% |
| `Forced Error: error caused due to pressure from the opponent’s aggressive or well-placed shot. e.g., rushed forehand sent long after a powerful serve or wide angle. The opponent pressured into an error.` | 5,041 | 10.01% |
| `Ace: is when the server scores in the first shot. The receiver didn’t touch the ball.` | 3,978 | 7.90% |
| `Double Fault: The server misses both first and second serves, resulting in an automatic point for the opponent.` | 1,956 | 3.88% |
| `Winner: When a shot (other than serve) lands in and isn’t touched or can’t be returned. Clean shot not returned (excluding net winners)` | 1,091 | 2.17% |

## `rules_knowledge_contextual`

- Fine category: **Rules Knowledge**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`derived`**
- Implementation type: **`derived_categorical`**
- Example source: annotation-triggered rules question bank
- Source fields: `rules_knowledge_contextual`
- Question template: `<rules_question_stem>`
- Why reasoning: Triggered by point annotations (serve outcome, score state, ball outcome, etc.). Tests general tennis-rules knowledge in a thematically related video context; the gold answer is the rule, not a video readout.

### Example 1

- Source: `annotation-triggered rules bank` / `rules_question_1`
- Question: According to the rules of singles tennis, where must a legal serve land for the serve to be valid?
  - (A) In the service box diagonally opposite the server
  - (B) Anywhere inside the opponent’s singles court
  - (C) In either service box on the opponent’s side of the net
  - (D) On or beyond the opponent’s baseline after the bounce
- Answer: (A) In the service box diagonally opposite the server

Eligibility is determined by annotation triggers rather than a direct source
field. See the [rules trigger map](../tennis_rules_trigger_map.md).

## `rules_knowledge_contextual_secondary`

- Fine category: **Rules Knowledge**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`derived`**
- Implementation type: **`derived_categorical`**
- Example source: annotation-triggered rules question bank
- Source fields: `rules_knowledge_contextual`
- Question template: `<rules_question_stem>`
- Why reasoning: Second matching rules-knowledge question for the same point when multiple triggers fire (capped by tennis_rules_trigger_map.max_per_point).

### Example 1

- Source: `annotation-triggered rules bank` / `rules_question_2`
- Question: If the server commits a fault on the first serve, what is the correct continuation of the point?
  - (A) The server immediately loses the point
  - (B) The server is awarded a second serve on the same point
  - (C) The receiver may choose whether the point is replayed
  - (D) The players change ends and the entire point is restarted
- Answer: (B) The server is awarded a second serve on the same point

Eligibility is determined by annotation triggers rather than a direct source
field. See the [rules trigger map](../tennis_rules_trigger_map.md).

# Audio Understanding

## `player_effort_sound_present`

- Fine category: **Impact and Play Sounds**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`derived`**
- Implementation type: **`derived_categorical`**
- Example source: latest generated categorical sample
- Source fields: `audio_cues`
- Question template: `Is a player grunt or effort sound audible?`

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_PlayerEffortSoundPresent_d2757a9f-1b8a-458f-80ae-5faa3659b846`
- Question: Is a player grunt or effort sound audible?
  - (A) No
  - (B) Yes
- Answer: (B) Yes

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **50,732** (100.00%)
- Expected generated data points: **50,732** (generation probability: 1.00)

**Correct-answer / composed-answer distribution among eligible points** (2 unique answers)

| Answer | Count | Percent |
|---|---:|---:|
| `Yes` | 34,364 | 67.74% |
| `No` | 16,368 | 32.26% |

**Source field `audio_cues` cue presence among eligible points** (425 unique cues; percentages are per-point presence rates and may sum to more than 100%)

| Value | Count | Percent |
|---|---:|---:|
| `thwack` | 47,928 | 94.47% |
| `grunt` | 32,674 | 64.41% |
| `squeak` | 28,837 | 56.84% |
| `umpire's call` | 16,852 | 33.22% |
| `cheer` | 13,190 | 26.00% |
| `thud` | 10,287 | 20.28% |
| `thump` | 8,900 | 17.54% |
| `clap` | 6,712 | 13.23% |
| `pop` | 5,825 | 11.48% |
| `net hit` | 4,325 | 8.53% |
| _Other cues (415 unique)_ | 39,682 point-cue occurrences | — |
| _Missing/unavailable_ | 69 | 0.14% |

## `court_movement_sound_present`

- Fine category: **Impact and Play Sounds**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`derived`**
- Implementation type: **`derived_categorical`**
- Example source: latest generated categorical sample
- Source fields: `audio_cues`
- Question template: `Are shoe squeaks audible during the point?`

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_CourtMovementSoundPresent_ead80da6-fe16-4008-967c-365f11608259`
- Question: Are shoe squeaks audible during the point?
  - (A) Yes
  - (B) No
- Answer: (B) No

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **50,732** (100.00%)
- Expected generated data points: **50,732** (generation probability: 1.00)

**Correct-answer / composed-answer distribution among eligible points** (2 unique answers)

| Answer | Count | Percent |
|---|---:|---:|
| `Yes` | 32,966 | 64.98% |
| `No` | 17,766 | 35.02% |

**Source field `audio_cues` cue presence among eligible points** (425 unique cues; percentages are per-point presence rates and may sum to more than 100%)

| Value | Count | Percent |
|---|---:|---:|
| `thwack` | 47,928 | 94.47% |
| `grunt` | 32,674 | 64.41% |
| `squeak` | 28,837 | 56.84% |
| `umpire's call` | 16,852 | 33.22% |
| `cheer` | 13,190 | 26.00% |
| `thud` | 10,287 | 20.28% |
| `thump` | 8,900 | 17.54% |
| `clap` | 6,712 | 13.23% |
| `pop` | 5,825 | 11.48% |
| `net hit` | 4,325 | 8.53% |
| _Other cues (415 unique)_ | 39,682 point-cue occurrences | — |
| _Missing/unavailable_ | 69 | 0.14% |

## `crowd_reaction_present`

- Fine category: **Crowd and Official Cues**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`derived`**
- Implementation type: **`derived_categorical`**
- Example source: latest generated categorical sample
- Source fields: `audio_cues`
- Question template: `Is there an audible crowd reaction?`

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_CrowdReactionPresent_ead80da6-fe16-4008-967c-365f11608259`
- Question: Is there an audible crowd reaction?
  - (A) No
  - (B) Yes
- Answer: (A) No

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **50,732** (100.00%)
- Expected generated data points: **50,732** (generation probability: 1.00)

**Correct-answer / composed-answer distribution among eligible points** (2 unique answers)

| Answer | Count | Percent |
|---|---:|---:|
| `No` | 33,573 | 66.18% |
| `Yes` | 17,159 | 33.82% |

**Source field `audio_cues` cue presence among eligible points** (425 unique cues; percentages are per-point presence rates and may sum to more than 100%)

| Value | Count | Percent |
|---|---:|---:|
| `thwack` | 47,928 | 94.47% |
| `grunt` | 32,674 | 64.41% |
| `squeak` | 28,837 | 56.84% |
| `umpire's call` | 16,852 | 33.22% |
| `cheer` | 13,190 | 26.00% |
| `thud` | 10,287 | 20.28% |
| `thump` | 8,900 | 17.54% |
| `clap` | 6,712 | 13.23% |
| `pop` | 5,825 | 11.48% |
| `net hit` | 4,325 | 8.53% |
| _Other cues (415 unique)_ | 39,682 point-cue occurrences | — |
| _Missing/unavailable_ | 69 | 0.14% |

## `official_call_present`

- Fine category: **Crowd and Official Cues**
- Format: **MCQ**
- Option space: **`closed`**
- Distractor generation: **`derived`**
- Implementation type: **`derived_categorical`**
- Example source: latest generated categorical sample
- Source fields: `audio_cues`
- Question template: `Is an umpire or fault call audible?`

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_OfficialCallPresent_d2757a9f-1b8a-458f-80ae-5faa3659b846`
- Question: Is an umpire or fault call audible?
  - (A) Yes
  - (B) No
- Answer: (B) No

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **50,732** (100.00%)
- Expected generated data points: **50,732** (generation probability: 1.00)

**Correct-answer / composed-answer distribution among eligible points** (2 unique answers)

| Answer | Count | Percent |
|---|---:|---:|
| `No` | 28,322 | 55.83% |
| `Yes` | 22,410 | 44.17% |

**Source field `audio_cues` cue presence among eligible points** (425 unique cues; percentages are per-point presence rates and may sum to more than 100%)

| Value | Count | Percent |
|---|---:|---:|
| `thwack` | 47,928 | 94.47% |
| `grunt` | 32,674 | 64.41% |
| `squeak` | 28,837 | 56.84% |
| `umpire's call` | 16,852 | 33.22% |
| `cheer` | 13,190 | 26.00% |
| `thud` | 10,287 | 20.28% |
| `thump` | 8,900 | 17.54% |
| `clap` | 6,712 | 13.23% |
| `pop` | 5,825 | 11.48% |
| `net hit` | 4,325 | 8.53% |
| _Other cues (415 unique)_ | 39,682 point-cue occurrences | — |
| _Missing/unavailable_ | 69 | 0.14% |

# Commentary & Reporting

## `point_caption`

- Fine category: **Point Narration**
- Format: **QA**
- Option space: **`open`**
- Distractor generation: **`none`**
- Implementation type: **`open_ended`**
- Example source: latest generated categorical sample
- Source fields: `point_caption`
- Question template: `What happened in this point? Provide a detailed caption. (<player1> - <player1_description>, <player2> - <player2_description>)`

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_Caption_ead80da6-fe16-4008-967c-365f11608259`
- Question: What happened in this point? Provide a detailed caption. (Jil Teichmann - "Jil Teichmann" is the player wearing a white cap, a black top, a black skirt, and white and tan shoes; Victoria Mboko - "Victoria Mboko" is the player wearing a white cap, white top, a green skirt, and white shoes.)
- Answer: The rally starts with the first serve by J. Teichmann that lands in the deuce service court. V. Mboko returns it with a backhand shot, to which J. Teichmann hits a forehand shot that lands the ball into the ad court of V. Mboko. V. Mboko is unable to return the shot. It is a winner by J. Teichman, giving her a point.

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **50,667** (99.87%)
- Expected generated data points: **50,667** (generation probability: 1.00)

**Source field `point_caption` distribution among eligible points** (50,081 unique values)

| Value | Count | Percent |
|---|---:|---:|
| _No non-missing values_ | 0 | 0.00% |
| _Other values (50,081 unique)_ | 50,667 | 100.00% |

## `how_point_ended_description`

- Fine category: **Ending Reporting**
- Format: **QA**
- Option space: **`open`**
- Distractor generation: **`none`**
- Implementation type: **`open_ended`**
- Example source: latest generated categorical sample
- Source fields: `how_point_ended_description`
- Question template: `Describe in detail how this point ended between <player1> and <player2>. (<player1> - <player1_description>, <player2> - <player2_description>)`

### Example 1

- Source: `9pHZXyxp8Jo` / `9pHZXyxp8Jo_PointEndDescription_ead80da6-fe16-4008-967c-365f11608259`
- Question: Describe in detail how this point ended between Jil Teichmann and Victoria Mboko. (Jil Teichmann - "Jil Teichmann" is the player wearing a white cap, a black top, a black skirt, and white and tan shoes; Victoria Mboko - "Victoria Mboko" is the player wearing a white cap, white top, a green skirt, and white shoes.)
- Answer: The point ended when J. Teichmann hit a winner.

### Complete-corpus statistics

- Total corpus points: **50,732**
- Eligible points: **50,666** (99.87%)
- Expected generated data points: **50,666** (generation probability: 1.00)

**Source field `how_point_ended_description` distribution among eligible points** (16,122 unique values)

| Value | Count | Percent |
|---|---:|---:|
| `The point ended when Djokovic hit the ball into the net.` | 368 | 0.73% |
| `The point ended when Federer hit the ball into the net.` | 360 | 0.71% |
| `The point ended when Nadal hit the ball into the net.` | 301 | 0.59% |
| `The point ended when Federer hit a winner.` | 255 | 0.50% |
| `The point ended when Djokovic hit a winner.` | 249 | 0.49% |
| `The point ended when Nadal hit a winner.` | 225 | 0.44% |
| `The point ended when Alcaraz hit the ball into the net.` | 211 | 0.42% |
| `The point ended when Sinner hit the ball into the net.` | 186 | 0.37% |
| `The point ended when Nadal hit the ball out of bounds.` | 175 | 0.35% |
| `The point ended when Alcaraz hit a winner.` | 154 | 0.30% |
| _Other values (16,112 unique)_ | 48,182 | 95.10% |

# Complete aggregated dataset totals

Source: `aggregated_data_complete`

- Video files: **252**
- Annotated points: **50,732**
- Eligible template–point pairs across all templates: **1,526,767**
- Expected generated data points across all templates: **1,452,666**

> The last two totals count a point once for every template it can produce. They are therefore larger than the number of unique annotated points.
