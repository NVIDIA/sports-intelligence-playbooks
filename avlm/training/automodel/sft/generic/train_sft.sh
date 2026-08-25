#!/usr/bin/env bash
# Run AutoModel SFT in an existing container without Slurm.
set -euo pipefail

_GENERIC_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -x /opt/venv/bin/automodel ]]; then
  echo "error: /opt/venv/bin/automodel not found — run inside nemo-automodel:26.06" >&2
  exit 1
fi

# shellcheck source=../../../../utils/_source_params.sh
source "${_GENERIC_DIR}/../../../../utils/_source_params.sh"

# shellcheck source=_train_lib.sh
source "${_GENERIC_DIR}/_train_lib.sh"
generic_init_launch
generic_train
