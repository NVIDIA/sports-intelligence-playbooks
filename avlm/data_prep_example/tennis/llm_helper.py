import json
import os
import time
import re

_TENNIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_env_files() -> None:
    """Load committed ``dev.env``, then gitignored ``dev_local.env`` overrides."""
    try:
        import dotenv
        tennis_dir = _TENNIS_DIR
        dotenv.load_dotenv(os.path.join(tennis_dir, "dev.env"))
        dotenv.load_dotenv(os.path.join(tennis_dir, "dev_local.env"), override=True)
        dotenv.load_dotenv()
    except ImportError:
        pass


_load_env_files()

# The OpenAI client is created lazily so this module can be imported without the
# ``openai`` package or an API key (e.g. when running the offline data-prep
# example, which never calls the LLM). The client is only built when an
# LLM-backed function is actually invoked.
_client = None

# GPT-5.2 on each provider (model slugs differ by endpoint).
OPENAI_API_BASE_URL = "https://api.openai.com/v1"
NVIDIA_INFERENCE_BASE_URL = "https://inference-api.nvidia.com"
OPENAI_DEFAULT_MODEL = "gpt-5.2"
NVIDIA_DEFAULT_MODEL = "azure/openai/gpt-5.2"


def _llm_settings() -> dict:
    """Read provider, credentials, and model from env (after loading dev*.env)."""
    _load_env_files()
    provider = os.getenv("LLM_API_PROVIDER", "nvidia").strip().lower()
    model_override = os.getenv("LLM_MODEL", "").strip()

    if provider == "openai":
        return {
            "provider": "openai",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_BASE_URL", OPENAI_API_BASE_URL),
            "model": model_override or OPENAI_DEFAULT_MODEL,
        }

    if provider not in ("nvidia", "nvidia_inference"):
        raise ValueError(
            f"Unsupported LLM_API_PROVIDER={provider!r}. "
            "Use 'nvidia' or 'openai'."
        )

    return {
        "provider": "nvidia",
        "api_key": os.getenv("NVIDIA_API_KEY"),
        "base_url": NVIDIA_INFERENCE_BASE_URL,
        "model": model_override or NVIDIA_DEFAULT_MODEL,
    }


def get_default_model() -> str:
    """Return the model name for the configured provider."""
    return _llm_settings()["model"]


def get_client():
    """Lazily create and cache the LLM client for the configured provider."""
    global _client
    if _client is not None:
        return _client

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The 'openai' package is required for LLM-backed generation.\n"
            "Install it with: pip install openai\n"
            "(Not needed for the offline data-prep example.)"
        ) from exc

    settings = _llm_settings()
    provider = settings["provider"]
    api_key = settings["api_key"]
    model = settings["model"]

    if provider == "openai":
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required when LLM_API_PROVIDER=openai.\n"
                "Set it in avlm/data_prep_example/tennis/dev_local.env or export OPENAI_API_KEY."
            )
        _client = OpenAI(
            timeout=120,
            api_key=api_key,
            base_url=settings["base_url"],
        )
        print(f"Using OpenAI API ({settings['base_url']}) with model: {model}")
    else:
        if not api_key:
            raise ValueError(
                "NVIDIA_API_KEY environment variable is required when LLM_API_PROVIDER=nvidia.\n"
                "Set it in avlm/data_prep_example/tennis/dev_local.env or export NVIDIA_API_KEY.\n"
                "Get your API key from: https://inference-api.nvidia.com"
            )
        _client = OpenAI(
            timeout=120,
            base_url=settings["base_url"],
            api_key=api_key,
        )
        print(f"Using NVIDIA Inference API with model: {model}")

    return _client

def generate_tennis_answer_from_llm(metadata: dict, field_value: str, prompt_template: str, model: str = None) -> str:
    """
    Generate an enhanced answer using LLM based on metadata and field value.
    
    Args:
        metadata: Dictionary containing point metadata and context
        field_value: Original field value
        prompt_template: Custom prompt template for answer generation
        model: LLM model to use (defaults to API-specific model)
        
    Returns:
        Generated answer string
    """
    if model is None:
        model = get_default_model()
    
    try:
        # Replace placeholders in prompt template
        prompt = prompt_template
        for key, value in metadata.items():
            # Key might already have <> brackets or not
            if key.startswith("<") and key.endswith(">"):
                placeholder = key
            else:
                placeholder = f"<{key}>"
            
            if placeholder in prompt and value is not None:
                prompt = prompt.replace(placeholder, str(value))
        
        api_params = {
            "model": model,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a professional tennis commentator and analyst. Generate detailed, accurate tennis commentary based on the provided metadata."
                },
                {
                    "role": "user", 
                    "content": f"Original caption: {field_value}\n\n{prompt}"
                }
            ],
            "temperature": 1,
            "max_tokens": 500,
        }
        
        response = get_client().chat.completions.create(**api_params)
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error generating LLM answer: {e}")
        return str(field_value)  # Fallback to original value


def generate_tennis_distractors_from_llm(metadata: dict, correct_answer: str, prompt_template: str, model: str = None) -> list:
    """
    Generate distractor options using LLM based on metadata and correct answer.
    
    Args:
        metadata: Dictionary containing point metadata and context
        correct_answer: The correct answer to generate distractors for
        prompt_template: Custom prompt template for distractor generation
        model: LLM model to use (defaults to API-specific model)
        
    Returns:
        List of distractor strings
    """
    if model is None:
        model = get_default_model()
    
    def _normalize_option(text: str) -> str:
        cleaned = text.strip().strip('"').strip("'")
        # Strip common bullet/letter/number prefixes
        cleaned = re.sub(r"^\s*[-*\u2022]\s*", "", cleaned)
        cleaned = re.sub(r"^\s*[A-Da-d]\)\s*", "", cleaned)
        cleaned = re.sub(r"^\s*\(?[A-Da-d]\)\s*", "", cleaned)
        cleaned = re.sub(r"^\s*\d+\.?\)?\s*", "", cleaned)
        return cleaned.strip()
    
    def _extract_distractors_from_text(text: str) -> list:
        # Try JSON first
        try:
            # Extract JSON object or array if present
            json_str = None
            if "{" in text and "}" in text:
                start = text.find("{")
                end = text.rfind("}")
                json_str = text[start:end+1]
            elif "[" in text and "]" in text:
                start = text.find("[")
                end = text.rfind("]")
                json_str = text[start:end+1]
            if json_str is not None:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and "distractors" in parsed and isinstance(parsed["distractors"], list):
                    return [str(_normalize_option(x)) for x in parsed["distractors"] if isinstance(x, str) and _normalize_option(x)]
                if isinstance(parsed, list):
                    return [str(_normalize_option(x)) for x in parsed if isinstance(x, str) and _normalize_option(x)]
        except Exception:
            pass
        
        # Fallback: split by lines
        result = []
        for raw_line in text.splitlines():
            line = _normalize_option(raw_line)
            if line:
                result.append(line)
        return result
    
    def _validate_and_clean(options: list) -> list:
        banned_keywords = [
            "unforced error on a routine shot",
            "unforced error",
            "net violation",
            "defensive lob",
            "lucky shot",
        ]
        seen = set()
        cleaned = []
        correct_norm = _normalize_option(correct_answer).lower()
        for opt in options:
            norm = _normalize_option(opt)
            lower = norm.lower()
            if not norm:
                continue
            if lower == correct_norm:
                continue
            if any(bad in lower for bad in banned_keywords):
                continue
            if lower in seen:
                continue
            seen.add(lower)
            cleaned.append(norm)
        return cleaned
    
    # Build prompt with metadata replacements
    try:
        prompt = prompt_template
        for key, value in metadata.items():
            if key.startswith("<") and key.endswith(">"):
                placeholder = key
            else:
                placeholder = f"<{key}>"
            if placeholder in prompt and value is not None:
                prompt = prompt.replace(placeholder, str(value))
        
        # Compose strict instruction
        players = []
        for k in ("player1", "player2"):
            v = metadata.get(k) or metadata.get(f"<{k}>")
            if v:
                players.append(str(v))
        players_hint = ", ".join(players) if players else "the players"
        
        system_msg = (
            "You are a tennis expert. Create plausible but incorrect tennis point descriptions. "
            "Return a JSON object with a 'distractors' array containing 4-6 short tennis descriptions."
        )
        user_msg_template = (
            "Correct answer: {correct}\n\n"
            "Create 4-6 plausible but incorrect tennis point descriptions. Focus on wrong winners, wrong shots, or wrong outcomes. "
            f"Use players {players_hint}. Keep descriptions under 120 characters each.\n\n"
            "Return JSON format:\n"
            "{{\n  \"distractors\": [\"description1\", \"description2\", \"description3\", \"description4\"]\n}}"
        )
        
        distractors = []
        max_attempts = 3
        for attempt in range(max_attempts):
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg_template.format(correct=correct_answer)},
            ]
            if attempt == 1:
                messages.append({
                    "role": "user",
                    "content": "Your previous output had fewer than 4 distinct distractors after filtering. Return 4-6 high-quality, distinct items now, JSON only."
                })
            elif attempt == 2:
                messages.append({
                    "role": "user",
                    "content": "Strict requirement: respond with a JSON object {\"distractors\": [ ... ]} containing 5-6 concise, distinct, high-quality incorrect options. No commentary."
                })
            
            api_params = {
                "model": model,
                "messages": messages,
                "temperature": 1,
            }
            
            response = get_client().chat.completions.create(**api_params)
            response_text = response.choices[0].message.content.strip()
            
            parsed = _extract_distractors_from_text(response_text)
            cleaned = _validate_and_clean(parsed)
            if len(cleaned) >= 3:
                distractors = [x for x in cleaned if x][:3]
                break
        # If still insufficient, attempt to augment from a small themed pool while avoiding banned/duplicates
        if len(distractors) < 3:
            
            fallback_pool = [
                "Shot landed beyond the singles sideline",
                "Ball sailed past the baseline under no pressure", 
                "Serve was called out but play continued",
                "Return floated long after a routine rally",
                "Ball hit into the net",
                "Shot went out of bounds", 
                "Player hit a winner",
                "Serve was an ace",
                "Ball landed in court",
                "Shot was hit wide",
                "Ball went long",
                "Player made an error",
                "Serve was a fault",
                "Ball clipped the net",
                "Shot landed short",
                "Player hit cross-court",
                "Ball went down the line",
                "Serve was good",
                "Return was successful"
            ]
            for cand in fallback_pool:
                if len(distractors) >= 3:
                    break
                cand_norm = _normalize_option(cand)
                if cand_norm and cand_norm.strip() and cand_norm.lower() != _normalize_option(correct_answer).lower() and cand_norm not in distractors:
                    distractors.append(cand_norm)
        
        # Final guard: if still short, keep legacy generics as last resort
        if len(distractors) < 3:
            generic_distractors = [
                "Player made an unforced error on a routine shot",
                "Point ended with a net violation",
                "Player won with a defensive lob",
            ]
            for distractor in generic_distractors:
                if len(distractors) < 3 and distractor and distractor.strip() and distractor not in distractors:
                    distractors.append(distractor)
        
        # Ensure we never return empty strings
        return [x for x in distractors if x and x.strip()][:3]
    
    except Exception as e:
        print(f"Error generating LLM distractors: {e}")
        # Return generic distractors as fallback
        return [
            "Player made an unforced error",
            "Point ended with a net violation", 
            "Player won with a lucky shot"
        ]


def generate_tennis_mcq_with_llm(metadata: dict, field_value: str, question_template: str, 
                                answer_prompt: str = None, distractor_prompt: str = None, 
                                model: str = None) -> tuple:
    """
    Generate complete MCQ (question, answer, distractors) using LLM.
    
    Args:
        metadata: Dictionary containing point metadata and context
        field_value: Original field value
        question_template: Question template with placeholders
        answer_prompt: Optional prompt for generating enhanced answer
        distractor_prompt: Optional prompt for generating distractors
        model: LLM model to use (defaults to API-specific model)
        
    Returns:
        Tuple of (question, correct_answer, distractors)
    """
    if model is None:
        model = get_default_model()
    # Generate enhanced answer if prompt provided, otherwise use original field value
    if answer_prompt and answer_prompt.strip():
        correct_answer = generate_tennis_answer_from_llm(metadata, field_value, answer_prompt, model)
    else:
        correct_answer = str(field_value)  # Use original answer when llm_answer_prompt is None or empty
    
    # Generate distractors if prompt provided
    if distractor_prompt:
        distractors = generate_tennis_distractors_from_llm(metadata, correct_answer, distractor_prompt, model)
    else:
        # Use default distractor generation
        distractors = [
            "Player made an unforced error",
            "Point ended with a net violation",
            "Player won with a lucky shot"
        ]
    
    # Replace placeholders in question template
    question = question_template
    for key, value in metadata.items():
        # Key might already have <> brackets or not
        if key.startswith("<") and key.endswith(">"):
            placeholder = key
        else:
            placeholder = f"<{key}>"
        
        if placeholder in question and value is not None:
            question = question.replace(placeholder, str(value))
    
    return question, correct_answer, distractors
