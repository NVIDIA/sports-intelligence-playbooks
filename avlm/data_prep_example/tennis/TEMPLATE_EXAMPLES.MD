# Legacy Template Examples

One generated question-and-answer example for each template in the default
legacy pipeline.

- MCQ config: `configs/generation/tennis_mcq_config.json`
- QA config: `configs/generation/tennis_qa_config.json`
- Example source: `training_tennis_data_output/mcq_qa_per_video`

# Match Facts

## `server_winner`

- Fine category: **Point Roles and Outcome**
- Format: **MCQ**
- Implementation type: **`categorical`**
- Source field: `server_winner`
- Question template: `Who served and who won this tennis point? (<player1> - <player1_description>, <player2> - <player2_description>)`

### Example 1

- Source: `sample_video_001` / `sample_video_001_ServerWinner_6f7b9d82-1744-47a2-afb9-91507d3c8a90`
- Question: Who served and who won this tennis point? (Jack Draper - "Jack Draper" is the player in the white t-shirt and white shorts. He is wearing a white cap and yellow shoes; Carlos Alcaraz - "Carlos Alcaraz" is the player in Gray and white t-shirt and gray pants. He is wearing yellow shoes and a white wrist band.)
  - (A) Carlos Alcaraz served and Carlos Alcaraz won
  - (B) Carlos Alcaraz served and Jack Draper won
  - (C) Jack Draper served and Jack Draper won
  - (D) Jack Draper served and Carlos Alcaraz won
- Answer: (A) Carlos Alcaraz served and Carlos Alcaraz won

## `receiver_winner`

- Fine category: **Point Roles and Outcome**
- Format: **MCQ**
- Implementation type: **`categorical`**
- Source field: `receiver_winner`
- Question template: `Who received the serve and who won this tennis point? (<player1> - <player1_description>, <player2> - <player2_description>)`

### Example 1

- Source: `sample_video_001` / `sample_video_001_ReceiverWinner_6f7b9d82-1744-47a2-afb9-91507d3c8a90`
- Question: Who received the serve and who won this tennis point? (Jack Draper - "Jack Draper" is the player in the white t-shirt and white shorts. He is wearing a white cap and yellow shoes; Carlos Alcaraz - "Carlos Alcaraz" is the player in Gray and white t-shirt and gray pants. He is wearing yellow shoes and a white wrist band.)
  - (A) Jack Draper received and Carlos Alcaraz won
  - (B) Carlos Alcaraz received and Jack Draper won
  - (C) Carlos Alcaraz received and Carlos Alcaraz won
  - (D) Jack Draper received and Jack Draper won
- Answer: (A) Jack Draper received and Carlos Alcaraz won

## `how_point_ended`

- Fine category: **Point Roles and Outcome**
- Format: **MCQ**
- Implementation type: **`categorical`**
- Source field: `how_point_ended`
- Question template: `How did this tennis point end?`

### Example 1

- Source: `sample_video_001` / `sample_video_001_PointEnd_6f7b9d82-1744-47a2-afb9-91507d3c8a90`
- Question: How did this tennis point end?
  - (A) Unforced Error: player makes a mistake during the point without significant pressure such as hitting the ball outside of the court (long or wide) or into the net during a routine shot. Routine miss without pressure.
  - (B) Violation: Rule-based loss of point includes player touching net, ball touches player, double bounce, foot fault, etc.
  - (C) Double Fault: The server misses both first and second serves, resulting in an automatic point for the opponent.
  - (D) Winner: When a shot (other than serve) lands in and isn’t touched or can’t be returned. Clean shot not returned (excluding net winners)
- Answer: (A) Unforced Error: player makes a mistake during the point without significant pressure such as hitting the ball outside of the court (long or wide) or into the net during a routine shot. Routine miss without pressure.

## `num_shots_exchanged`

- Fine category: **Rally Structure**
- Format: **MCQ**
- Implementation type: **`numeric_range`**
- Source field: `num_shots_exchanged`
- Question template: `How many shots were exchanged in this rally?`

### Example 1

- Source: `sample_video_001` / `sample_video_001_RallyLength_6f7b9d82-1744-47a2-afb9-91507d3c8a90`
- Question: How many shots were exchanged in this rally?
  - (A) 2 shots
  - (B) 6 shots
  - (C) 21+ shots (Extended rally)
  - (D) 1 shot
- Answer: (A) 2 shots

## `num_serving_attempts_until_successful`

- Fine category: **Score and Serve State**
- Format: **MCQ**
- Implementation type: **`categorical`**
- Source field: `num_serving_attempts_until_successful`
- Question template: `How many serving attempts until successful?`

### Example 1

- Source: `sample_video_001` / `sample_video_001_ServingAttempts_6f7b9d82-1744-47a2-afb9-91507d3c8a90`
- Question: How many serving attempts until successful?
  - (A) 2
  - (B) 3
  - (C) 1
- Answer: (C) 1

## `score_after`

- Fine category: **Score and Serve State**
- Format: **MCQ**
- Implementation type: **`freeform_with_distractors`**
- Source field: `score_after`
- Question template: `<score_before>. What was the score after this point? (<player1> - <player1_description>, <player2> - <player2_description>)`

### Example 1

- Source: `sample_video_001` / `sample_video_001_ScoreAfter_6f7b9d82-1744-47a2-afb9-91507d3c8a90`
- Question: Draper vs. Alcaraz:{0-0, 0-0}. What was the score after this point? (Jack Draper - "Jack Draper" is the player in the white t-shirt and white shorts. He is wearing a white cap and yellow shoes; Carlos Alcaraz - "Carlos Alcaraz" is the player in Gray and white t-shirt and gray pants. He is wearing yellow shoes and a white wrist band.)
  - (A) Draper vs. Alcaraz:{0-0, 15-0}
  - (B) Draper vs. Alcaraz:{0-0, 15-15}
  - (C) Draper vs. Alcaraz:{0-0, 0-15}
  - (D) Draper vs. Alcaraz:{0-0, 0-0}
- Answer: (C) Draper vs. Alcaraz:{0-0, 0-15}

# Commentary & Reporting

## `mcq_how_point_ended_description_mechanism_mismatch`

- Fine category: **Ending Reporting**
- Format: **MCQ**
- Implementation type: **`llm_gen_distractor`**
- Source field: `how_point_ended_description`
- Question template: `Describe in detail how this point ended between <player1> and <player2>. (<player1> - <player1_description>, <player2> - <player2_description>)`

### Example 1

- Source: `sample_video_001` / `sample_video_001_McqHowPointEndedDescriptionMechanismMismatch_6f7b9d82-1744-47a2-afb9-91507d3c8a90`
- Question: Describe in detail how this point ended between Jack Draper and Carlos Alcaraz. (Jack Draper - "Jack Draper" is the player in the white t-shirt and white shorts. He is wearing a white cap and yellow shoes; Carlos Alcaraz - "Carlos Alcaraz" is the player in Gray and white t-shirt and gray pants. He is wearing yellow shoes and a white wrist band.)
  - (A) Point ended with a net violation
  - (B) Player won with a lucky shot
  - (C) Player made an unforced error
  - (D) The point ended when Draper hit the ball into his own court with a forehand shot.
- Answer: (D) The point ended when Draper hit the ball into his own court with a forehand shot.

## `mcq_how_point_ended_description_player_misattribution`

- Fine category: **Ending Reporting**
- Format: **MCQ**
- Implementation type: **`llm_gen_distractor`**
- Source field: `how_point_ended_description`
- Question template: `Describe in detail how this point ended between <player1> and <player2>. (<player1> - <player1_description>, <player2> - <player2_description>)`

### Example 1

- Source: `sample_video_001` / `sample_video_001_McqHowPointEndedDescriptionPlayerMisattribution_6f7b9d82-1744-47a2-afb9-91507d3c8a90`
- Question: Describe in detail how this point ended between Jack Draper and Carlos Alcaraz. (Jack Draper - "Jack Draper" is the player in the white t-shirt and white shorts. He is wearing a white cap and yellow shoes; Carlos Alcaraz - "Carlos Alcaraz" is the player in Gray and white t-shirt and gray pants. He is wearing yellow shoes and a white wrist band.)
  - (A) The point ended when Draper hit the ball into his own court with a forehand shot.
  - (B) Point ended with a net violation
  - (C) Player won with a lucky shot
  - (D) Player made an unforced error
- Answer: (A) The point ended when Draper hit the ball into his own court with a forehand shot.

## `mcq_how_point_ended_description_location_mismatch`

- Fine category: **Ending Reporting**
- Format: **MCQ**
- Implementation type: **`llm_gen_distractor`**
- Source field: `how_point_ended_description`
- Question template: `Describe in detail how this point ended between <player1> and <player2>. (<player1> - <player1_description>, <player2> - <player2_description>)`

### Example 1

- Source: `sample_video_001` / `sample_video_001_McqHowPointEndedDescriptionLocationMismatch_6f7b9d82-1744-47a2-afb9-91507d3c8a90`
- Question: Describe in detail how this point ended between Jack Draper and Carlos Alcaraz. (Jack Draper - "Jack Draper" is the player in the white t-shirt and white shorts. He is wearing a white cap and yellow shoes; Carlos Alcaraz - "Carlos Alcaraz" is the player in Gray and white t-shirt and gray pants. He is wearing yellow shoes and a white wrist band.)
  - (A) Player made an unforced error
  - (B) The point ended when Draper hit the ball into his own court with a forehand shot.
  - (C) Point ended with a net violation
  - (D) Player won with a lucky shot
- Answer: (B) The point ended when Draper hit the ball into his own court with a forehand shot.

## `mcq_how_point_ended_description_unsupported_embellishments`

- Fine category: **Ending Reporting**
- Format: **MCQ**
- Implementation type: **`llm_gen_distractor`**
- Source field: `how_point_ended_description`
- Question template: `Describe in detail how this point ended between <player1> and <player2>. (<player1> - <player1_description>, <player2> - <player2_description>)`

### Example 1

- Source: `sample_video_001` / `sample_video_001_McqHowPointEndedDescriptionUnsupportedEmbellishments_6f7b9d82-1744-47a2-afb9-91507d3c8a90`
- Question: Describe in detail how this point ended between Jack Draper and Carlos Alcaraz. (Jack Draper - "Jack Draper" is the player in the white t-shirt and white shorts. He is wearing a white cap and yellow shoes; Carlos Alcaraz - "Carlos Alcaraz" is the player in Gray and white t-shirt and gray pants. He is wearing yellow shoes and a white wrist band.)
  - (A) Point ended with a net violation
  - (B) Player won with a lucky shot
  - (C) The point ended when Draper hit the ball into his own court with a forehand shot.
  - (D) Player made an unforced error
- Answer: (C) The point ended when Draper hit the ball into his own court with a forehand shot.

## `mcq_caption_mechanism_mismatch`

- Fine category: **Point Narration**
- Format: **MCQ**
- Implementation type: **`llm_gen_distractor`**
- Source field: `point_caption`
- Question template: `What happened in this point? Provide a detailed caption. (<player1> - <player1_description>, <player2> - <player2_description>)`

### Example 1

- Source: `sample_video_001` / `sample_video_001_McqCaptionMechanismMismatch_6f7b9d82-1744-47a2-afb9-91507d3c8a90`
- Question: What happened in this point? Provide a detailed caption. (Jack Draper - "Jack Draper" is the player in the white t-shirt and white shorts. He is wearing a white cap and yellow shoes; Carlos Alcaraz - "Carlos Alcaraz" is the player in Gray and white t-shirt and gray pants. He is wearing yellow shoes and a white wrist band.)
  - (A) Point ended with a net violation
  - (B) Player made an unforced error
  - (C) Alcaraz delivers a well-placed serve, followed by Draper hitting the ball into his own court, conceding the point to Alcaraz. The crowd starts clapping after the point.
  - (D) Player won with a lucky shot
- Answer: (C) Alcaraz delivers a well-placed serve, followed by Draper hitting the ball into his own court, conceding the point to Alcaraz. The crowd starts clapping after the point.

## `mcq_caption_player_misattribution`

- Fine category: **Point Narration**
- Format: **MCQ**
- Implementation type: **`llm_gen_distractor`**
- Source field: `point_caption`
- Question template: `What happened in this point? Provide a detailed caption. (<player1> - <player1_description>, <player2> - <player2_description>)`

### Example 1

- Source: `sample_video_001` / `sample_video_001_McqCaptionPlayerMisattribution_6f7b9d82-1744-47a2-afb9-91507d3c8a90`
- Question: What happened in this point? Provide a detailed caption. (Jack Draper - "Jack Draper" is the player in the white t-shirt and white shorts. He is wearing a white cap and yellow shoes; Carlos Alcaraz - "Carlos Alcaraz" is the player in Gray and white t-shirt and gray pants. He is wearing yellow shoes and a white wrist band.)
  - (A) Point ended with a net violation
  - (B) Player won with a lucky shot
  - (C) Alcaraz delivers a well-placed serve, followed by Draper hitting the ball into his own court, conceding the point to Alcaraz. The crowd starts clapping after the point.
  - (D) Player made an unforced error
- Answer: (C) Alcaraz delivers a well-placed serve, followed by Draper hitting the ball into his own court, conceding the point to Alcaraz. The crowd starts clapping after the point.

## `mcq_caption_location_mismatch`

- Fine category: **Point Narration**
- Format: **MCQ**
- Implementation type: **`llm_gen_distractor`**
- Source field: `point_caption`
- Question template: `What happened in this point? Provide a detailed caption. (<player1> - <player1_description>, <player2> - <player2_description>)`

### Example 1

- Source: `sample_video_001` / `sample_video_001_McqCaptionLocationMismatch_6f7b9d82-1744-47a2-afb9-91507d3c8a90`
- Question: What happened in this point? Provide a detailed caption. (Jack Draper - "Jack Draper" is the player in the white t-shirt and white shorts. He is wearing a white cap and yellow shoes; Carlos Alcaraz - "Carlos Alcaraz" is the player in Gray and white t-shirt and gray pants. He is wearing yellow shoes and a white wrist band.)
  - (A) Player won with a lucky shot
  - (B) Player made an unforced error
  - (C) Point ended with a net violation
  - (D) Alcaraz delivers a well-placed serve, followed by Draper hitting the ball into his own court, conceding the point to Alcaraz. The crowd starts clapping after the point.
- Answer: (D) Alcaraz delivers a well-placed serve, followed by Draper hitting the ball into his own court, conceding the point to Alcaraz. The crowd starts clapping after the point.

## `mcq_caption_unsupported_embellishments`

- Fine category: **Point Narration**
- Format: **MCQ**
- Implementation type: **`llm_gen_distractor`**
- Source field: `point_caption`
- Question template: `What happened in this point? Provide a detailed caption. (<player1> - <player1_description>, <player2> - <player2_description>)`

### Example 1

- Source: `sample_video_001` / `sample_video_001_McqCaptionUnsupportedEmbellishments_6f7b9d82-1744-47a2-afb9-91507d3c8a90`
- Question: What happened in this point? Provide a detailed caption. (Jack Draper - "Jack Draper" is the player in the white t-shirt and white shorts. He is wearing a white cap and yellow shoes; Carlos Alcaraz - "Carlos Alcaraz" is the player in Gray and white t-shirt and gray pants. He is wearing yellow shoes and a white wrist band.)
  - (A) Alcaraz delivers a well-placed serve, followed by Draper hitting the ball into his own court, conceding the point to Alcaraz. The crowd starts clapping after the point.
  - (B) Player won with a lucky shot
  - (C) Player made an unforced error
  - (D) Point ended with a net violation
- Answer: (A) Alcaraz delivers a well-placed serve, followed by Draper hitting the ball into his own court, conceding the point to Alcaraz. The crowd starts clapping after the point.

## `how_point_ended_description`

- Fine category: **Ending Reporting**
- Format: **QA**
- Implementation type: **`open_ended`**
- Source field: `how_point_ended_description`
- Question template: `Describe in detail how this point ended between <player1> and <player2>. (<player1> - <player1_description>, <player2> - <player2_description>)`

### Example 1

- Source: `sample_video_001` / `sample_video_001_PointEndDescription_6f7b9d82-1744-47a2-afb9-91507d3c8a90`
- Question: Describe in detail how this point ended between Jack Draper and Carlos Alcaraz. (Jack Draper - "Jack Draper" is the player in the white t-shirt and white shorts. He is wearing a white cap and yellow shoes; Carlos Alcaraz - "Carlos Alcaraz" is the player in Gray and white t-shirt and gray pants. He is wearing yellow shoes and a white wrist band.)
- Answer: The point ended when Draper hit the ball into his own court with a forehand shot.

## `point_caption`

- Fine category: **Point Narration**
- Format: **QA**
- Implementation type: **`open_ended`**
- Source field: `point_caption`
- Question template: `What happened in this point? Provide a detailed caption. (<player1> - <player1_description>, <player2> - <player2_description>)`

### Example 1

- Source: `sample_video_001` / `sample_video_001_Caption_6f7b9d82-1744-47a2-afb9-91507d3c8a90`
- Question: What happened in this point? Provide a detailed caption. (Jack Draper - "Jack Draper" is the player in the white t-shirt and white shorts. He is wearing a white cap and yellow shoes; Carlos Alcaraz - "Carlos Alcaraz" is the player in Gray and white t-shirt and gray pants. He is wearing yellow shoes and a white wrist band.)
- Answer: Alcaraz delivers a well-placed serve, followed by Draper hitting the ball into his own court, conceding the point to Alcaraz. The crowd starts clapping after the point.

# Fine-Grained Visual Recognition

## `mcq_caption_action_mismatch`

- Fine category: **Action Consistency**
- Format: **MCQ**
- Implementation type: **`llm_gen_distractor`**
- Source field: `point_caption`
- Question template: `What happened in this point? Provide a detailed caption including the action of the players. (<player1> - <player1_description>, <player2> - <player2_description>)`

### Example 1

- Source: `sample_video_001` / `sample_video_001_McqCaptionActionMismatch_6f7b9d82-1744-47a2-afb9-91507d3c8a90`
- Question: What happened in this point? Provide a detailed caption including the action of the players. (Jack Draper - "Jack Draper" is the player in the white t-shirt and white shorts. He is wearing a white cap and yellow shoes; Carlos Alcaraz - "Carlos Alcaraz" is the player in Gray and white t-shirt and gray pants. He is wearing yellow shoes and a white wrist band.)
  - (A) Player won with a lucky shot
  - (B) Point ended with a net violation
  - (C) Alcaraz delivers a well-placed serve, followed by Draper hitting the ball into his own court, conceding the point to Alcaraz. The crowd starts clapping after the point.
  - (D) Player made an unforced error
- Answer: (C) Alcaraz delivers a well-placed serve, followed by Draper hitting the ball into his own court, conceding the point to Alcaraz. The crowd starts clapping after the point.

# Audio Understanding

## `audio_cues`

- Fine category: **Audible Events**
- Format: **QA**
- Implementation type: **`open_ended`**
- Source field: `audio_cues`
- Question template: `what audio cues or sounds can be heard during this point`

### Example 1

- Source: `sample_video_001` / `sample_video_001_AudioCues_6f7b9d82-1744-47a2-afb9-91507d3c8a90`
- Question: what audio cues or sounds can be heard during this point
- Answer: Whacks, squeaks, grunt, claps.
