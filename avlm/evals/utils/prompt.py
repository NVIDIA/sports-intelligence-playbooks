# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

DEFAULT_SYSTEM_PROMPT = (
        "You are an expert VLM evaluator. Provide accurate, fair, and detailed scoring "
        "based on the given criteria."
    )

USER_PROMPT = f"""Your task is to compare the model prediction directly against the conversation and ground truth, and use the provided metadata as authoritative context for verification and disambiguation. Prioritize agreement with gt, and use metadata only for direct contradictions or clear disambiguation. Do not invent contradictions from ambiguous metadata fields. Treat unambiguous player aliases as the same player (for example, Amritraj and Riske-Amritraj) when the conversation or gt makes the mapping clear. For first serve/double fault/ace serve-related claims, use metadata fields like 'num_serving_attempts_until_successful' as supporting context, but do not use that field to penalize a fault/let/then-successful-serve sequence when gt and pred agree on that sequence and metadata does not explicitly state a different complete serve sequence. If the prediction adds a specific point-ending category (like Ace, Double Fault, Winner, etc) that is not mentioned in the gt, do NOT penalize unless this addition is directly contradicted by the metadata. If metadata supports or is consistent with the pred's extra detail, do not treat it as a hallucination. However, if pred adds a point-ending type that is directly contradicted by metadata, penalize accordingly.

If no metadata is provided, score purely on pred vs gt and be strict about extra unsupported details.

IMPORTANT: Respond only with valid JSON. The "score" field must be an integer from 1 to 10, where 10 is best, never 0.

**Ground Truth**
REPLACE_GT

**Conversation**
REPLACE_CONVERSATION

**Prediction**
REPLACE_PREDICTION

**Metadata**
REPLACE_METADATA

**Evaluation Guidance:**
- Primary: Similarity to gt (semantic and factual)
- Secondary: Consistency with metadata when present; penalize claims directly contradicted by available metadata
- **MCQ / single-letter answers**: If gt is of the form "(A) ...", "(B) ...", "(C) ...", "(D) ..." and pred is only the letter (e.g. "A", "B", "C", "D") or "A." / "The answer is A.", treat as a **full match (score 10 ** when the letter matches the gt choice. Do not penalize for missing the full "(X) ..." text.
- Treat unambiguous player aliases as equivalent, including surname/full-name/hyphenated variants such as "Amritraj" and "Riske-Amritraj"
- For serve/double fault/ace claims, check metadata fields like 'num_serving_attempts_until_successful' as supporting context; do not let that field override matching gt/pred fault/let/then-successful-serve descriptions unless metadata explicitly gives a different complete serve sequence
- If pred adds a point-ending type (Ace, Double Fault, etc) not in gt, check metadata:
    - If metadata supports or is consistent, do NOT penalize
    - If metadata directly contradicts, penalize
    - If metadata is absent, be strict and penalize unsupported additions
- Ignore plausible but unsupported details that are not grounded in gt/metadata

**Scoring (1-10, integers):**
- 10: Pred essentially matches gt; no contradictions with metadata
- 9: Very strong match; at most tiny omissions
- 8: Minor omissions or wording differences; consistent with metadata
- 7: Captures main idea; some errors/unsupported details
- 6: Partially relevant; noticeable errors/contradictions
- 1-5: Poor match or contradicted by metadata

Respond ONLY with a JSON object of the form:
{{
  "score": <1-10>,
  "reasoning": "<brief, precise justification>",
  "strengths": ["<strength 1>", "<strength 2>"],
  "weaknesses": ["<weakness 1>", "<weakness 2>"],
  "metadata_alignment": "<note on consistency with metadata>"
}}
"""

def get_default_system_prompt():
    return DEFAULT_SYSTEM_PROMPT

def get_user_prompt(gt, conversation, prediction, metadata, user_prompt=None):
    prompt = USER_PROMPT.replace("REPLACE_GT", gt)
    prompt = prompt.replace("REPLACE_CONVERSATION", conversation)
    prompt = prompt.replace("REPLACE_PREDICTION", prediction)
    prompt = prompt.replace("REPLACE_METADATA", metadata if metadata else "")
    if user_prompt:
        prompt += f"\n\n**Additional Evaluation Guidance (takes precedence):**\n{user_prompt}"

    return prompt
