Data Curation
=============

This section documents how to annotate sports video data and convert those annotations into training datasets for multimodal language models. It covers annotation guidelines, the data preparation pipeline, and where to find example code in the `sports_intelligence <https://gitlab-master.nvidia.com/avlm/sports_intelligence>`_ repository under `avlm/ <https://gitlab-master.nvidia.com/avlm/sports_intelligence/-/tree/main/avlm>`_. The workflow below uses **tennis** as the worked example, but the same curation principles—structured event labels, dense captions, and template-driven Q&A generation—apply to other sports with sport-specific schemas and templates.

Example Tennis Annotation Guideline
------------------------------------

The following is a **tennis-specific** annotation guideline that we used for multimodal language models. Other sports will use different fields and taxonomies, but the overall approach—segmenting events, capturing outcomes and narratives, and enforcing quality control—carries over directly.

`Tennis Annotation Guideline (PDF) <../Tennis_Annotation_Guideline.pdf>`_ — a step-by-step guide for point-by-point annotation of full tennis match videos, including game context, segment timestamps, dense captions, and quality-control workflows.

From Tennis Annotation to Training Data
---------------------------------------

After match videos are annotated following the guideline above, each match is exported as one JSON file per video (for example, `1D8sm1NCwUs.json <https://gitlab-master.nvidia.com/avlm/sports_intelligence/-/blob/main/avlm/data_prep_example/tennis/raw_tennis_data_input/1D8sm1NCwUs.json>`_). Those files list match metadata (player names, segment bounds) and a ``table`` of point-level rows: server/receiver, winner, scores, how the point ended, play-by-play captions, audio cues, per-point clip paths, and related fields.

The tennis data-prep pipeline below shows how to turn that structure into supervised (video, question, answer) examples suitable for supervised fine-tuning (SFT). The reference implementation lives in `avlm/data_prep_example/tennis/ <https://gitlab-master.nvidia.com/avlm/sports_intelligence/-/tree/main/avlm/data_prep_example/tennis>`_ in the `sports_intelligence <https://gitlab-master.nvidia.com/avlm/sports_intelligence>`_ repo. You can adapt the same pattern for other sports by defining sport-specific templates and configs.

Pipeline Overview
~~~~~~~~~~~~~~~~~

.. code-block:: text

   raw_tennis_data_input/<videoId>.json
         │  generate_mcq_qa.py   (templates in configs/)
         ▼
   training_tennis_data_output/mcq_qa_per_video/<videoId>_derived_mcq_qa.json
         │  split_dataset.py     (point-level split)
         ▼
   training_tennis_data_output/data_splits_hf/tennis_mcq_qa_{train,validation,test_seen_videos,test_unseen_videos}.jsonl

Each annotated point becomes many training rows: one row per question template (20 templates in total — 17 MCQ and 3 open-ended QA).

What the Generator Does
~~~~~~~~~~~~~~~~~~~~~~~

``generate_mcq_qa.py`` walks every point in the raw JSON and applies **named templates** from ``configs/``:

- ``tennis_mcq_config.json`` — multiple-choice questions (server/winner, score, rally length, how the point ended, …)
- ``tennis_qa_config.json`` — open-ended QA (point caption, how the point ended, audio cues)
- ``question_types.json`` — optional allow-list of template names (omit to generate all)

Each template specifies which annotation **field** supplies the ground-truth answer, the **question** text (with placeholders such as ``<player1>`` and ``<score_before>`` filled from metadata), and a **type** that controls how options are built:

- **categorical** — the template defines a fixed list of multiple-choice options in the config (for example, the four server/winner combinations, or the set of “how the point ended” categories). The generator reads the annotated field, picks the matching option as the correct answer, and presents the full predefined list as the MCQ choices.
- **numeric_range** — the annotated field is a number (for example, ``num_shots_exchanged``). The template maps numeric intervals to labeled answers (for example, 1 → “1 shot”, 2 → “2 shots”, 11–20 → “11–20 shots”). The generator finds which interval contains the annotated value and uses the corresponding label as the correct answer; the other interval labels are the distractors.
- **open_ended** — assistant reply is the annotation field verbatim
- **llm_gen_distractor** — MCQ where the correct option is still the annotated text, but three wrong options are written by an LLM (mcq_* templates only)

For each point the generator: (1) resolves placeholders, (2) picks or derives the correct answer, (3) shuffles MCQ options, and (4) emits a HuggingFace **conversation** record with the point's relative ``video`` path, question/answer text, and a ``class`` field set to the template name (for example ``server_winner`` or ``score_after``).

Most templates are **fully deterministic** and need no LLM. Only the ``mcq_*`` distractor templates call an API (``openai`` + ``python-dotenv``; API key in a local ``dev_local.env``).

Train / Validation / Test Splits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``split_dataset.py`` assigns **points** (not individual questions) to splits so every Q&A derived from the same point stays together:

- **train** / **validation** — points from seen matches
- **test_seen_videos** — held-out points from a seen match (generalization within a known game)
- **test_unseen_videos** — points from a fully held-out match (generalization to new players/context)

Output Format
~~~~~~~~~~~~~

Final artifacts are JSONL files under ``training_tennis_data_output/data_splits_hf/``. Each line is one (video, question) pair. The ``class`` field carries the question-type / template name the row was generated from (for example ``server_winner``, ``score_after``); eval tooling uses it to route MCQ vs descriptive scoring (see :doc:`evaluation`).

.. code-block:: json

   {
     "id": "1D8sm1NCwUs_ServerWinner_09c90920-...",
     "class": "server_winner",
     "conversation": [
       {
         "role": "user",
         "content": [
           {"type": "video", "path": "1D8sm1NCwUs/1D8sm1NCwUs_seg_01_13_19_timeoffset_end_3.mp4"},
           {"type": "text", "text": "Who served and who won...?\n(A) ...\n(B) ...\n(C) ...\n(D) ..."}
         ]
       },
       {
         "role": "assistant",
         "content": [{"type": "text", "text": "(A) Carlos Alcaraz served and Carlos Alcaraz won"}]
       }
     ]
   }

Open-ended QA rows omit the ``(A)/(B)/...`` options; the assistant message is the annotation field directly.

For ``server_winner`` and ``receiver_winner`` templates, MCQ options list all four server/receiver × winner player-name combinations (for example ``Jack Draper served and Carlos Alcaraz won``), and the correct answer names both participants explicitly.

Question Templates (Summary)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Category
     - Templates
   * - Deterministic MCQ
     - ``server_winner``, ``receiver_winner``, ``how_point_ended``, ``num_shots_exchanged``, ``num_serving_attempts_until_successful``, ``score_after``
   * - LLM MCQ (``mcq_*``)
     - Distractor variants on ``how_point_ended_description`` and ``point_caption``: generic rewrites, mechanism mismatch, player misattribution, location mismatch, unsupported embellishments, action mismatch
   * - Open-ended QA
     - ``how_point_ended_description``, ``point_caption``, ``audio_cues``

Full template names and descriptions are listed in the tennis README linked in the next section.

Example Code and Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **README (start here):** `avlm/data_prep_example/tennis/README.md <https://gitlab-master.nvidia.com/avlm/sports_intelligence/-/blob/main/avlm/data_prep_example/tennis/README.md>`_
- **End-to-end runner:** `prepare_training_data_pipeline.py <https://gitlab-master.nvidia.com/avlm/sports_intelligence/-/blob/main/avlm/data_prep_example/tennis/prepare_training_data_pipeline.py>`_
- **MCQ/QA generation:** `generate_mcq_qa.py <https://gitlab-master.nvidia.com/avlm/sports_intelligence/-/blob/main/avlm/data_prep_example/tennis/generate_mcq_qa.py>`_
- **Point-level splits:** `split_dataset.py <https://gitlab-master.nvidia.com/avlm/sports_intelligence/-/blob/main/avlm/data_prep_example/tennis/split_dataset.py>`_
- **Templates:** `configs/ <https://gitlab-master.nvidia.com/avlm/sports_intelligence/-/tree/main/avlm/data_prep_example/tennis/configs>`_

The example ships with **sample annotation JSON for two matches (20 points)** under ``raw_tennis_data_input/``. Training video files are not bundled — only the annotation JSON and generated JSONL.

Quick Start
~~~~~~~~~~~

From the tennis example directory in a clone of `sports_intelligence <https://gitlab-master.nvidia.com/avlm/sports_intelligence>`_:

.. code-block:: bash

   cd avlm/data_prep_example/tennis
   uv sync && source .venv/bin/activate   # only needed for mcq_* LLM distractors
   python prepare_training_data_pipeline.py

The pipeline writes its output to ``training_tennis_data_output/``; the final train, validation, and test splits are in ``training_tennis_data_output/data_splits_hf/*.jsonl``.
