#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Greedy video chat against a running vLLM server (local file:// URLs).
#
#   bash avlm/deployment/vLLM/client/chat_video_completion.sh
#     every MCQ, then the tennis demo caption (temperature=0 top_p=1 top_k=1)
#   VLLM_CHAT_PRESET=mcq|caption|all
#   VLLM_CHAT_MESSAGE='Custom question' bash ...
set -euo pipefail

_VLLM_MCQ_LABELS=()
_VLLM_MCQ_QUESTIONS=()
_vllm_add_mcq() {
  _VLLM_MCQ_LABELS+=("$1")
  local question
  question="$(cat)"
  _VLLM_MCQ_QUESTIONS+=("${question}")
}

_vllm_add_mcq "MCQ: serving attempts" <<'EOF'
How many serving attempts were made in this point?
(A) 3
(B) 4
(C) 1
(D) 2
EOF

_vllm_add_mcq "MCQ: server location" <<'EOF'
Was the server on the near or far side of the visible court?
(A) far-court
(B) near-court
EOF

_vllm_add_mcq "MCQ: receiver location" <<'EOF'
Was the receiver on the near or far side of the visible court?
(A) near-court
(B) far-court
EOF

_vllm_add_mcq "MCQ: score after the point" <<'EOF'
Major vs. Mustermann:{6-4, 5-3, 30-15}. What was the score after this point?
(A) Major vs. Mustermann:{6-4, 5-3, 40-30}
(B) Major vs. Mustermann:{6-4, 5-3, 40-15}
(C) Major vs. Mustermann:{6-4, 5-3, 30-30}
(D) Major vs. Mustermann:{6-4, 5-3, 30-15}
EOF

_vllm_add_mcq "MCQ: losing player's lateral movement" <<'EOF'
What was the losing player's lateral movement?
(A) right to left
(B) left to right
(C) no lateral movement
EOF

_vllm_add_mcq "MCQ: losing player's depth movement" <<'EOF'
How did the losing player move forward or backward?
(A) backward
(B) forward
(C) no depth movement
EOF

_vllm_add_mcq "MCQ: server and winner" <<'EOF'
Who served and who won this tennis point?
(A) Mustermann served and Mustermann won
(B) Major served and Mustermann won
(C) Mustermann served and Major won
(D) Major served and Major won
EOF

_vllm_add_mcq "MCQ: winner's role" <<'EOF'
Did the server or receiver win the point?
(A) receiver
(B) server
EOF

read -r -d '' _VLLM_DEFAULT_QUESTION_CAPTION <<'EOF' || true
What happened in this point? Provide a detailed caption. (Major - "Major" is the player in the white sleeveless tennis dress with a pleated skirt. She is wearing white shoes and has dark hair tied back in a ponytail.; Mustermann - "Mustermann" is the player in the black sleeveless top and black pleated skirt with white piping along the sides and hem. She is wearing white shoes and has blonde hair tied back in a ponytail. She is holding a green and black tennis racket with a red grip.)
EOF

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${DEPLOY_ROOT}/../../.." && pwd)"

# shellcheck source=../utils/_serve_params.sh
source "${DEPLOY_ROOT}/utils/_serve_params.sh"
source_serve_params "${DEPLOY_ROOT}" "${REPO_ROOT}"

VIDEO="${VLLM_CHAT_VIDEO:-assets/tennis_demo_video/full_tennis_point.mp4}"
if [[ "${VIDEO}" != /* ]]; then
  VIDEO="${REPO_ROOT}/${VIDEO}"
fi
[[ -f "${VIDEO}" ]] || {
  echo "error: video not found: ${VIDEO}" >&2
  exit 2
}

BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:${VLLM_PORT:-12500}}"
if ! curl -sf "${BASE_URL}/health" >/dev/null; then
  echo "error: vLLM not healthy at ${BASE_URL} — run: bash avlm/deployment/vLLM/vllm_server.sh start" >&2
  exit 1
fi
MODEL="${VLLM_CHAT_MODEL:-${SERVED_MODEL_NAME:-nemotron_3_nano_omni}}"
MAX_TOKENS="${VLLM_CHAT_MAX_TOKENS:-${VLLM_DEFAULT_MAX_TOKENS:-512}}"
# Match scripts/hf_upload_download_inference/inference/video_inference.py default.
MAX_TOKENS_CAPTION="${VLLM_CHAT_MAX_TOKENS_CAPTION:-256}"
STREAM="${VLLM_CHAT_STREAM:-0}"
# Greedy decode (HF video_inference uses do_sample=False).
VLLM_CHAT_TEMPERATURE="${VLLM_CHAT_TEMPERATURE:-0}"
VLLM_CHAT_TOP_P="${VLLM_CHAT_TOP_P:-1}"
VLLM_CHAT_TOP_K="${VLLM_CHAT_TOP_K:-1}"
# Set to 1 only for MP4s with a vLLM-compatible audio track (demo clip is video-only).
VLLM_CHAT_USE_AUDIO_IN_VIDEO="${VLLM_CHAT_USE_AUDIO_IN_VIDEO:-0}"

ENABLE_THINKING="${VLLM_CHAT_ENABLE_THINKING:-}"
if [[ -z "${ENABLE_THINKING}" ]]; then
  if _vllm_truthy "${VLLM_ENABLE_REASONING:-0}"; then
    ENABLE_THINKING=1
  else
    ENABLE_THINKING=0
  fi
fi

_vllm_build_payload() {
  local message="$1"
  local max_tokens="$2"
  VIDEO_PATH="${VIDEO}" MESSAGE="${message}" MODEL="${MODEL}" MAX_TOKENS="${max_tokens}" \
    STREAM="${STREAM}" ENABLE_THINKING="${ENABLE_THINKING}" \
    TEMPERATURE="${VLLM_CHAT_TEMPERATURE}" TOP_P="${VLLM_CHAT_TOP_P}" TOP_K="${VLLM_CHAT_TOP_K}" \
    USE_AUDIO_IN_VIDEO="${VLLM_CHAT_USE_AUDIO_IN_VIDEO}" python3 <<'PY'
import json
import os
from pathlib import Path

def truthy(s: str) -> bool:
    return s.strip().lower() in {"1", "true", "yes", "on"}

video = Path(os.environ["VIDEO_PATH"]).resolve()
message = os.environ["MESSAGE"]
model = os.environ["MODEL"]
max_tokens = int(os.environ["MAX_TOKENS"])
stream = truthy(os.environ["STREAM"])
enable_thinking = truthy(os.environ["ENABLE_THINKING"])
temperature = float(os.environ["TEMPERATURE"])
top_p = float(os.environ["TOP_P"])
top_k = int(os.environ["TOP_K"])
use_audio = truthy(os.environ["USE_AUDIO_IN_VIDEO"])
# Hub / video_inference.py: one user string "<video>" + question (no newline). Nemotron
# chat_template.jinja inserts "<video>\\n" before text when only video_url + plain text are
# sent; include "<video>" in text so the template skips the extra newline (see jinja
# counters.videos reset when "<video>" in text).
video_token = os.environ.get("VLLM_VIDEO_TOKEN", "<video>")
user_text = message if message.startswith(video_token) else f"{video_token}{message}"

payload = {
    "model": model,
    "max_tokens": max_tokens,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video.as_uri()}},
                {"type": "text", "text": user_text},
            ],
        }
    ],
    "chat_template_kwargs": {"enable_thinking": enable_thinking},
    "mm_processor_kwargs": {"use_audio_in_video": use_audio},
    "temperature": temperature,
    "top_p": top_p,
    "top_k": top_k,
}
if stream:
    payload["stream"] = True
print(json.dumps(payload))
PY
}

_vllm_print_chat_response() {
  local body_file="$1"
  BODY_FILE="${body_file}" python3 <<'PY'
import json
import os
import sys

path = os.environ["BODY_FILE"]
raw = open(path, encoding="utf-8").read()
if not raw.strip():
    print("error: empty response from vLLM (is the server running?)", file=sys.stderr)
    sys.exit(1)

def from_sse(lines):
    parts = []
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        parts.append(delta.get("content") or "")
    return "".join(parts).strip()

if raw.lstrip().startswith("data:"):
    text = from_sse(raw.splitlines())
else:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        print("error: response is not JSON:", raw[:500], file=sys.stderr)
        sys.exit(1)
    if obj.get("error"):
        print("error:", obj["error"], file=sys.stderr)
        sys.exit(1)
    msg = (obj.get("choices") or [{}])[0].get("message") or {}
    text = (msg.get("content") or "").strip()

print(text)
PY
}

_VLLM_REQUEST_LABELS=()
_VLLM_REQUEST_SECONDS=()
_VLLM_LAST_SECONDS=""

_vllm_curl_chat() {
  local payload="$1"
  local resp http_code curl_meta curl_flags=(-sS)
  resp="$(mktemp)"
  if _vllm_truthy "${STREAM}"; then
    curl_flags+=(-N)
  fi
  curl_meta="$(curl "${curl_flags[@]}" -o "${resp}" -w '%{http_code} %{time_total}' \
    "${BASE_URL}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "${payload}")" || {
    echo "[ERROR] curl failed (exit $?)" >&2
    rm -f "${resp}"
    return 1
  }
  http_code="${curl_meta%% *}"
  _VLLM_LAST_SECONDS="${curl_meta##* }"
  if [[ "${http_code}" != "200" ]]; then
    echo "[ERROR] HTTP ${http_code} from vLLM:" >&2
    cat "${resp}" >&2
    echo >&2
    rm -f "${resp}"
    return 1
  fi
  _vllm_print_chat_response "${resp}"
  rm -f "${resp}"
}

_vllm_run_question() {
  local label="$1"
  local message="$2"
  local max_tokens="$3"
  echo "[INFO] POST ${BASE_URL}/v1/chat/completions video=${VIDEO}" >&2
  echo >&2
  echo "[INFO] ${label} (max_tokens=${max_tokens})" >&2
  echo
  echo "Input prompt:"
  echo "${message}"
  echo
  echo "Output response:"
  _vllm_curl_chat "$(_vllm_build_payload "${message}" "${max_tokens}")"
  _VLLM_REQUEST_LABELS+=("${label}")
  _VLLM_REQUEST_SECONDS+=("${_VLLM_LAST_SECONDS}")
  echo
  printf 'Time: %.3fs\n' "${_VLLM_LAST_SECONDS}"
  echo
}

_vllm_print_timing_summary() {
  ((${#_VLLM_REQUEST_SECONDS[@]})) || return 0
  echo
  echo "--------"
  echo
  local i
  for ((i = 0; i < ${#_VLLM_REQUEST_SECONDS[@]}; i++)); do
    printf '%s\t%s\n' "${_VLLM_REQUEST_LABELS[i]}" "${_VLLM_REQUEST_SECONDS[i]}"
  done | python3 -c '
import sys
rows_in = [line.rstrip("\n").split("\t", 1) for line in sys.stdin if line.strip()]
vals = [float(seconds) for _label, seconds in rows_in]
n = len(vals)
runs = [("run", "question", "time")]
runs += [(str(i), label, f"{float(seconds):.3f}s") for i, (label, seconds) in enumerate(rows_in, 1)]
stats = [
    ("n", "average", "min", "max"),
    (str(n), f"{sum(vals)/n:.3f}s", f"{min(vals):.3f}s", f"{max(vals):.3f}s"),
]

def write_table(rows):
    widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell.ljust(widths[col]) for col, cell in enumerate(row)))

print("Request Processing Time Table")
write_table(runs)
print()
write_table(stats)
'
}

_run_mcqs() {
  local i
  for ((i = 0; i < ${#_VLLM_MCQ_QUESTIONS[@]}; i++)); do
    if ((i > 0)); then
      echo
      echo "--------"
      echo
    fi
    _vllm_run_question "${_VLLM_MCQ_LABELS[i]}" "${_VLLM_MCQ_QUESTIONS[i]}" "${MAX_TOKENS}"
  done
}

_run_preset() {
  local preset="$1"
  case "${preset}" in
    mcq)
      _run_mcqs
      ;;
    caption)
      _vllm_run_question "Detailed caption" "${_VLLM_DEFAULT_QUESTION_CAPTION}" "${MAX_TOKENS_CAPTION}"
      ;;
    all)
      _run_mcqs
      echo
      echo "--------"
      echo
      _vllm_run_question "Detailed caption" "${_VLLM_DEFAULT_QUESTION_CAPTION}" "${MAX_TOKENS_CAPTION}"
      ;;
    *)
      echo "error: VLLM_CHAT_PRESET must be mcq, caption, or all (got: ${preset})" >&2
      exit 2
      ;;
  esac
}

if [[ -n "${VLLM_CHAT_MESSAGE:-}" ]]; then
  _vllm_run_question "Custom question" "${VLLM_CHAT_MESSAGE}" "${MAX_TOKENS}"
elif ((${#@} > 0)); then
  _vllm_run_question "Custom question" "$*" "${MAX_TOKENS}"
else
  _run_preset "${VLLM_CHAT_PRESET:-all}"
fi
_vllm_print_timing_summary
