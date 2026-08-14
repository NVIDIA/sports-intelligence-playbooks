Evaluation
==========

After inference writes predictions, we score them with two complementary methods:

- **MCQ accuracy** — letter matching for multiple-choice questions (fast, deterministic)
- **LLM-as-judge** — rubric scoring for open-ended captions and descriptions where wording varies but factual alignment matters

Both approaches apply across sports. Code lives in the `sports_intelligence <https://gitlab-master.nvidia.com/avlm/sports_intelligence>`_ repo under ``avlm/evals/``; the shipped tennis question classes and prompts are a reference implementation, and the same patterns extend to other domains with sport-specific templates, metadata schemas, and judge rubrics. Predictions must exist first at:

.. code-block:: text

   avlm/inference/outputs/<INFERENCE_NAME>/predictions.jsonl

Each line is one (video, question) pair. See :doc:`inference/index` for generating predictions.

Two Evaluation Modes
--------------------

Routing is driven by the question **type** on each prediction (for example ``server_winner`` or ``point_caption``). In curated datasets this type is stored on each row as the ``class`` field (see :doc:`data_curation`). Types are registered as either **MCQ** or **descriptive** in ``avlm/evals/utils/question_class.py``:

- **MCQ** — scored locally by letter matching; no external API calls
- **Descriptive** — scored by an LLM judge; requires a judge API key

When building a new sport, register question types in that file and add matching templates in your data-prep configs (see :doc:`data_curation`). MCQ templates format answers as ``(A) ...``, ``(B) ...``; open-ended templates use free-text ground truth suitable for LLM judging.

Tennis reference classes
~~~~~~~~~~~~~~~~~~~~~~~~

Our tennis eval set includes MCQ and descriptive classes such as:

.. list-table::
   :header-rows: 1
   :widths: 38 22 40

   * - Class (tennis)
     - Eval method
     - Task type
   * - ``score_after``, ``server_winner``, ``receiver_winner``
     - MCQ
     - Structured outcome labels (score, participants)
   * - ``how_point_ended``, ``num_serving_attempts_until_successful``, ``num_shots_exchanged``
     - MCQ
     - Categorical or bucketed event attributes
   * - ``how_point_ended_description``, ``point_caption``
     - LLM judge
     - Open-ended narrative (what happened, play-by-play)

MCQ Evaluation
--------------

The MCQ scorer reads inference predictions and compares each model answer to the annotated correct letter. The headline metric is the **percentage of correct answers**.

Answer format
~~~~~~~~~~~~~

Answers are expected in the form ``(B) <option text>``. The scorer extracts the leading letter from both the model output and the reference answer. If the model reply is free text or missing the ``(A)``–``(D)`` prefix, it is counted as a **parse failure** and excluded from the accuracy denominator. Rows with malformed reference answers are skipped.

Results
~~~~~~~

Results are written to ``mcq/mcq_results.json`` under the inference output directory. The summary includes:

- **Accuracy** — share of parsed predictions that matched the correct letter
- **Counts** — how many were correct, incorrect, unparseable, or missing because inference failed
- **By question type** — the same accuracy and counts for each MCQ template (for example who served and won, how the point ended)

Run MCQ eval
~~~~~~~~~~~~

.. code-block:: bash

   INFERENCE_DIR=avlm/inference/outputs/<INFERENCE_NAME> \
   bash avlm/evals/scripts/run_mcq_eval.sh

Output:

.. code-block:: text

   avlm/inference/outputs/<INFERENCE_NAME>/mcq/mcq_results.json

LLM-as-Judge Evaluation
-----------------------

The LLM judge scores **descriptive** question types. A separate model rates each answer against the reference on a **1–10** scale and returns a short justification plus notes on factual alignment.

Use LLM judging when there is no single letter or exact string to match — for example, event captions or play-by-play descriptions where phrasing can differ but facts must hold.

Full event context for the judge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The judge does not see only the reference answer and the model reply. For each example it also receives:

1. **Reference answer** — the annotation from the eval set
2. **Original prompt** — the full question the VLM was asked (including the video reference)
3. **Model answer** — the generated text (any internal reasoning trace is stripped before scoring)
4. **Event metadata** — the full annotated row for that clip from preprocessed source JSON

Why include event metadata?
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Descriptive ground truth is often a **short caption or play-by-play string**. The model answer may use different wording but still be correct — or it may add **extra factual claims** that are not spelled out in the reference (for example, whether the point ended on an ace, a double fault, or a winner, or whether a serve was first or second).

Comparing prediction to ground truth alone is not enough in those cases:

- **Ground truth alone** cannot confirm or refute every fine-grained claim the model makes.
- **Strict pred-vs-GT matching** would unfairly penalize valid extra detail that the annotator simply did not repeat in the short reference string.
- **Lenient semantic matching** would let plausible but wrong details slip through.

**Event metadata** breaks that tradeoff. It is the structured annotation row for that point — the same fields used during data curation (see :doc:`data_curation`): server/receiver, scores, ``how_point_ended``, ``num_serving_attempts_until_successful``, rally length, captions, and related labels. The judge treats metadata as **authoritative context** to:

- **Verify** specific claims in the prediction (for example, cross-check serve-related statements against ``num_serving_attempts_until_successful``).
- **Disambiguate** point-ending categories (ace, double fault, winner, unforced error, and so on) when the reference caption is vague or abbreviated.
- **Avoid false hallucination penalties** when the model adds a detail that ground truth omitted but metadata supports.
- **Penalize contradictions** when the prediction conflicts with structured fields, even if the wording sounds plausible.

If metadata is omitted, the rubric falls back to strict pred-vs-GT scoring and penalizes unsupported additions. When metadata is present, scoring is primarily grounded-truth alignment, with metadata used as a secondary consistency check.

How metadata is attached
^^^^^^^^^^^^^^^^^^^^^^^^

For each prediction, the scorer loads the per-video annotation file from the configured metadata directory, finds the matching event row (keyed by id), and injects it into the judge message. Other sports follow the same pattern: one JSON file per video, event rows keyed by id.

Metadata can be included in the user message, the system message, or left out entirely (configurable in ``default.yaml``). The shipped tennis config passes the full event row in the user message alongside ground truth, conversation, and prediction.

Judge prompt
~~~~~~~~~~~~

The rubric lives in ``avlm/evals/utils/prompt.py``. At runtime the scorer fills in the reference answer, original prompt, model answer, and event metadata. An optional domain-specific paragraph from ``system_prompt`` in ``default.yaml`` is appended to the system message (the shipped config adds tennis adjudicator guidance).

**System message** (default):

.. code-block:: text

   You are an expert VLM evaluator. Provide accurate, fair, and detailed scoring
   based on the given criteria.

   You are also an expert tennis commentator and rules adjudicator. Interpret tennis terminology precisely 
   (serve, return, rally, winner, forced/unforced error, ace, double fault, volley, down-the-line, cross-court, baseline). 
   When metadata includes score flow or serve attempts, penalize only direct contradictions; 
   do not overrule matching GT/pred serve-sequence descriptions from ambiguous attempt-count fields. 
   Prefer concise, technical justifications.


**User message** (template; ``REPLACE_*`` slots are filled per example):

.. code-block:: text

   Your task is to compare the model prediction directly against the conversation
   and ground truth, and use the provided metadata as authoritative context for
   verification and disambiguation. Prioritize agreement with gt, and use metadata
   only for direct contradictions or clear disambiguation. Do not invent
   contradictions from ambiguous metadata fields. Treat unambiguous player aliases
   as the same player (for example, Amritraj and Riske-Amritraj) when the
   conversation or gt makes the mapping clear. For first serve/double fault/ace
   serve-related claims, use metadata fields like
   'num_serving_attempts_until_successful' as supporting context, but do not use
   that field to penalize a fault/let/then-successful-serve sequence when gt and
   pred agree on that sequence and metadata does not explicitly state a different
   complete serve sequence. If the prediction adds a specific point-ending
   category (like Ace, Double Fault, Winner, etc) that is not mentioned in the gt,
   do NOT penalize unless this addition is directly contradicted by the metadata.
   If metadata supports or is consistent with the pred's extra detail, do not
   treat it as a hallucination. However, if pred adds a point-ending type that is
   directly contradicted by metadata, penalize accordingly.

   If no metadata is provided, score purely on pred vs gt and be strict about
   extra unsupported details.

   IMPORTANT: Respond only with valid JSON. The "score" field must be an integer
   from 1 to 10, where 10 is best, never 0.

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
   - Secondary: Consistency with metadata when present; penalize claims directly
     contradicted by available metadata
   - **MCQ / single-letter answers**: If gt is of the form "(A) ...", "(B) ...",
     "(C) ...", "(D) ..." and pred is only the letter (e.g. "A", "B", "C", "D")
     or "A." / "The answer is A.", treat as a **full match (score 10 ** when the
     letter matches the gt choice. Do not penalize for missing the full "(X) ..."
     text.
   - Treat unambiguous player aliases as equivalent, including
     surname/full-name/hyphenated variants such as "Amritraj" and
     "Riske-Amritraj"
   - For serve/double fault/ace claims, check metadata fields like
     'num_serving_attempts_until_successful' as supporting context; do not let that
     field override matching gt/pred fault/let/then-successful-serve descriptions
     unless metadata explicitly gives a different complete serve sequence
   - If pred adds a point-ending type (Ace, Double Fault, etc) not in gt, check
     metadata:
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
   {
     "score": <1-10>,
     "reasoning": "<brief, precise justification>",
     "strengths": ["<strength 1>", "<strength 2>"],
     "weaknesses": ["<weakness 1>", "<weakness 2>"],
     "metadata_alignment": "<note on consistency with metadata>"
   }

The tennis examples in the prompt (serve attempts, point-ending categories) illustrate how metadata cross-check works; adapt ``prompt.py`` and the config ``system_prompt`` for other sports.

Judge configuration
~~~~~~~~~~~~~~~~~~~

Copy and edit ``avlm/evals/qa_llm_judge/default.yaml``. You need:

- An OpenAI-compatible **API base URL** and **judge model** name
- A **metadata directory** of per-video annotation JSON files

Optional settings cover parallelism, timeouts, how much metadata to include, and domain-specific system guidance.

Set the API key before running:

.. code-block:: bash

   export CLIENT_API_KEY=<YOUR_CLIENT_API_KEY>

LLM judge requires ``openai`` and ``pyyaml``. Install in base Python or a venv; if using a venv, set ``JUDGE_VENV_PATH`` to its root.

Run LLM judge
~~~~~~~~~~~~~

.. code-block:: bash

   INFERENCE_DIR=avlm/inference/outputs/<INFERENCE_NAME> \
   bash avlm/evals/scripts/run_qa_llm_judge.sh

With config override and sample limit:

.. code-block:: bash

   CONFIG=avlm/evals/qa_llm_judge/default.yaml \
   MAX_LLM_JUDGE_SAMPLES=100 \
   INFERENCE_DIR=avlm/inference/outputs/<INFERENCE_NAME> \
   bash avlm/evals/scripts/run_qa_llm_judge.sh

Output:

.. code-block:: text

   avlm/inference/outputs/<INFERENCE_NAME>/llm_judge/llm_judge_predictions.json

This file includes per-example scores and justifications, plus run-level statistics (mean, median, score distribution).

LLM judge summary
~~~~~~~~~~~~~~~~~

A second step rolls per-example judge output into a compact summary (overall and broken down by question type):

.. code-block:: bash

   INFERENCE_DIR=avlm/inference/outputs/<INFERENCE_NAME> \
   bash avlm/evals/scripts/run_qa_llm_judge_eval.sh

Output:

.. code-block:: text

   avlm/inference/outputs/<INFERENCE_NAME>/llm_judge/llm_judge_results.json

Outputs Layout
--------------

.. code-block:: text

   avlm/inference/outputs/<INFERENCE_NAME>/
   ├── predictions.jsonl          # from inference
   ├── mcq/
   │   └── mcq_results.json       # MCQ accuracy
   └── llm_judge/
       ├── llm_judge_predictions.json   # per-entry judge scores
       └── llm_judge_results.json       # aggregated judge metrics

Inference + Eval Pipeline
-------------------------

``avlm/inference/common/scripts/run_inference_eval_pipeline.sh`` runs inference, then MCQ scoring and LLM-judge stages end to end. Set ``INFERENCE_BACKEND`` to ``automodel`` or ``megatron_bridge``. Use ``STAGES`` to run a subset (for example ``mcq,judge,judge_eval``) or point at an existing ``INFERENCE_DIR`` to score predictions without re-running inference.

.. code-block:: bash

   INFERENCE_BACKEND=automodel \
   INFERENCE_CONFIG=avlm/inference/automodel/configs/default_sft.yaml \
   bash avlm/inference/common/scripts/run_inference_eval_pipeline.sh

Score existing predictions only:

.. code-block:: bash

   STAGES=mcq,judge,judge_eval \
   INFERENCE_DIR=avlm/inference/outputs/<INFERENCE_NAME> \
   bash avlm/inference/common/scripts/run_inference_eval_pipeline.sh

See `avlm/inference/README.md <https://gitlab-master.nvidia.com/avlm/sports_intelligence/-/blob/main/avlm/inference/README.md>`_ for pipeline overrides and eval-suite examples, and `avlm/evals/README.MD <https://gitlab-master.nvidia.com/avlm/sports_intelligence/-/blob/main/avlm/evals/README.MD>`_ for script-level reference.
