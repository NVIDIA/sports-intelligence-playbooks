#!/usr/bin/env bash
# Submit one Slurm automodel LoRA job per trial in a hyperparameter-search YAML.
#
#   HYPERPARAM_CONFIG=avlm/training/hyperparam_search/configs/automodel_lora/lora_hyperparam_search_trials.yaml \
#     bash avlm/training/hyperparam_search/automodel_lora_sbatch_starter.sh
#
# Inline environment overrides: num_nodes=8 WANDB_MODE=online hyperparam_name=...
# Override precedence: inline environment > hyperparameter-search YAML > launch_local.yaml > inherited shell exports.
set -euo pipefail

script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hyperparam_mode=automodel_lora
avlm_utils_dir="${script_dir}/../../utils"
slurm_dir="$(cd -- "${script_dir}/../automodel/lora/slurm" && pwd)"
sbatch_starter="${slurm_dir}/sbatch/sbatch_starter.sh"

# shellcheck source=../../utils/_source_params.sh
source "${avlm_utils_dir}/_source_params.sh"

: "${HYPERPARAM_CONFIG:?Set HYPERPARAM_CONFIG=path/to/hyperparam_search.yaml}"

export CLUSTER_PARAMS="${CLUSTER_PARAMS:-${slurm_dir}/launch_local.yaml}"
[[ "${CLUSTER_PARAMS}" == /* ]] || CLUSTER_PARAMS="${slurm_dir}/${CLUSTER_PARAMS#"${slurm_dir}"/}"
export CLUSTER_PARAMS
training_cli_commit_env_overrides "${slurm_dir}" \
  "${SBATCH_CLI_OVERRIDE_KEYS[@]}" "${SBATCH_ENV_ONLY_KEYS[@]}" hyperparam_name
source_cluster_params "${slurm_dir}"
resolve_avlm_repo_roots_from_mode_dir "${slurm_dir}"

hyperparam_config="${HYPERPARAM_CONFIG}"
[[ "${hyperparam_config}" == /* ]] || hyperparam_config="${REPO_ROOT}/${hyperparam_config}"
if [[ ! -f "${hyperparam_config}" ]]; then
  echo "error: hyperparameter config not found: ${hyperparam_config}" >&2
  exit 1
fi

_py=/opt/venv/bin/python3
[[ -x "${_py}" ]] || _py=$(command -v python3)

hyperparam_exports="$("${_py}" "${avlm_utils_dir}/_load_params_yaml.py" "${hyperparam_config}" --hyperparam-defaults)"
apply_yaml_exports_preserving_cli "${hyperparam_exports}"

_hyperparam_name_yaml="$("${_py}" -c "
import yaml, sys
cfg = yaml.safe_load(open(sys.argv[1])) or {}
print((cfg.get('hyperparam_name') or '').strip())
" "${hyperparam_config}")"
hyperparam_name="${hyperparam_name:-${_hyperparam_name_yaml}}"
: "${hyperparam_name:?Set hyperparam_name in the hyperparameter-search YAML or inline environment}"

outputs_hyperparam_base="${script_dir}/outputs/${hyperparam_mode}/${hyperparam_name}"
logs_hyperparam_base="${script_dir}/logs/${hyperparam_mode}/${hyperparam_name}"

_hyperparam_build_submit_args() {
  local -n _out=$1
  _out=()
  local key
  for key in "${SBATCH_CLI_OVERRIDE_KEYS[@]}"; do
    case "${key}" in
      MODEL_NAME|CONFIG_YAML|CONFIG_YAML_REL|OUTPUT_BASE|LOGS_DIR) continue ;;
    esac
    if [[ -n "${!key+x}" ]]; then
      _out+=("${key}=${!key}")
    fi
  done
}

trial_idx=0
while IFS=$'\t' read -r trial_name trial_config; do
  model_name="${hyperparam_name}_${trial_name}"
  run_ts="$(date +%Y%m%d_%H%M%S)"
  logs_dir="${logs_hyperparam_base}/${trial_name}/run_${run_ts}"

  submit_args=()
  _hyperparam_build_submit_args submit_args
  if [[ "${trial_config}" == /* ]]; then
    submit_args+=( CONFIG_YAML="${trial_config}" )
  else
    submit_args+=( CONFIG_YAML_REL="${trial_config}" )
  fi
  submit_args+=(
    MODEL_NAME="${model_name}"
    OUTPUT_BASE="${outputs_hyperparam_base}"
    LOGS_DIR="${logs_dir}"
  )

  ((trial_idx++)) || true
  echo "Submitting trial ${trial_idx}: ${trial_name} (${model_name})"
  env _AVLM_PIN_INHERITED_ENV=1 "${submit_args[@]}" bash "${sbatch_starter}"
done < <("${_py}" "${avlm_utils_dir}/_load_params_yaml.py" "${hyperparam_config}" --hyperparam-trials)

if (( trial_idx == 0 )); then
  echo "error: no trials found in ${hyperparam_config}" >&2
  exit 1
fi
