#!/usr/bin/env bash
# Build a portable deep_ep wheel for NeMo AutoModel (nemo-automodel:26.06-style images).
#
# Run INSIDE the nemo-automodel container (CUDA toolkit + nvcc required):
#   bash wheels/deepep/building/build_deepep_wheel.sh
#
# Default: all supported arches (8.0 8.6 8.9 + 9.0 10.0 12.0) as two wheels:
#   bash wheels/deepep/building/build_deepep_wheel.sh
#
# Pre-Hopper only (SM 8.0 / 8.6 / 8.9 — faster, intranode kernels):
#   DEEPEP_WHEEL_PROFILE=pre-hopper-sm80-sm89 bash wheels/deepep/building/build_deepep_wheel.sh
#
# Post-Hopper only (SM 9.0 / 10.0 / 12.0 — matches NGC Dockerfile):
#   DEEPEP_WHEEL_PROFILE=post-hopper-sm90-sm100-sm120 bash wheels/deepep/building/build_deepep_wheel.sh
#
# From the login node (non-interactive Slurm job):
#   bash wheels/deepep/building/run_build_in_container.sh
#
# After loading a container on any node, install the wheel from lustre:
#   bash wheels/deepep/install_deepep_wheel.sh
#
# Output: deep_ep-*.whl in wheels/deepep/; deepep_wheel_build_manifest.txt in this directory.
# Docs: wheels/deepep/DEEPEP.MD
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPEP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST_FILE="${SCRIPT_DIR}/deepep_wheel_build_manifest.txt"
# shellcheck source=../../../avlm/utils/_source_params.sh
source "${SCRIPT_DIR}/../../../avlm/utils/_source_params.sh"
resolve_avlm_repo_roots_from_avlm_root "$(cd "${SCRIPT_DIR}/../../../avlm" && pwd)"
PROJECT_ROOT="${REPO_ROOT}"
OUTPUT_DIR="${DEEPEP_WHEEL_OUTPUT_DIR:-${DEEPEP_DIR}}"

# Match docker/Dockerfile + uv.lock (hybrid-ep branch).
DEEPEP_COMMIT="${DEEPEP_COMMIT:-7febc6e25660af0f54d95dd781ecdcd62265ecca}"
DEEPEP_PATCH="${DEEPEP_PATCH:-/opt/Automodel/docker/common/deepep.patch}"
RDMA_CORE_TAG="${RDMA_CORE_TAG:-v60.0}"
NVSHMEM_VERSION="${NVSHMEM_VERSION:-3.4.5}"

# pre-hopper-sm80-sm89: SM 8.0 8.6 8.9 — intranode MoE dispatch (no NVSHMEM internode kernels).
# post-hopper-sm90-sm100-sm120: SM 9.0 + 10.0 (Blackwell) + 12.0 — full HybridEP multinode.
# all: both wheels (covers 8.0–12.0; cannot be one wheel at this DeepEP revision).
# Legacy aliases: ampere, hopper, post-hopper-sm90-sm120 (old name without explicit sm100).
DEEPEP_WHEEL_PROFILE="${DEEPEP_WHEEL_PROFILE:-all}"
DEEPEP_MIN_CUDA_ARCH_MAJOR=8
# Must match docker/Dockerfile TORCH_CUDA_ARCH_LIST for DeepEP (include 10.0 for Blackwell / SM 10).
DEEPEP_POST_HOPPER_CUDA_ARCH_LIST="${DEEPEP_POST_HOPPER_CUDA_ARCH_LIST:-9.0 10.0 12.0}"
DEEPEP_PRE_HOPPER_CUDA_ARCH_LIST="${DEEPEP_PRE_HOPPER_CUDA_ARCH_LIST:-8.0 8.6 8.9}"

BUILD_ROOT="${DEEPEP_BUILD_ROOT:-/tmp/deepep_wheel_build_$$}"
KEEP_BUILD_DIR="${DEEPEP_KEEP_BUILD_DIR:-0}"
MAX_JOBS="${MAX_JOBS:-8}"

log() {
  echo "[deepep-wheel $(date -Iseconds)] $*"
}

die() {
  echo "[deepep-wheel ERROR] $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1 (run inside nemo-automodel container)"
}

cleanup() {
  if [[ "${KEEP_BUILD_DIR}" == "1" ]]; then
    log "keeping build tree at ${BUILD_ROOT}"
    return
  fi
  if [[ -d "${BUILD_ROOT}" ]]; then
    log "removing build tree ${BUILD_ROOT}"
    rm -rf "${BUILD_ROOT}"
  fi
}
trap cleanup EXIT

# DeepEP setup.py to_nvcc_gencode() only accepts entries like "8.0" — not "8.0+PTX".
normalize_cuda_arch_list() {
  local raw="${1}"
  local normalized=""
  local part stripped major
  local IFS=' ,;'
  read -ra parts <<< "${raw}"
  for part in "${parts[@]}"; do
    [[ -n "${part}" ]] || continue
    stripped="${part%%+PTX}"
    stripped="${stripped%%+ptx}"
    if [[ ! "${stripped}" =~ ^[0-9]+\.[0-9]+[A-Za-z]?$ ]]; then
      die "invalid CUDA arch entry: ${part} (DeepEP expects forms like 8.0 or 9.0, not ${part})"
    fi
    major="${stripped%%.*}"
    if (( major < DEEPEP_MIN_CUDA_ARCH_MAJOR )); then
      log "warning: skipping ${stripped} (DeepEP requires SM ${DEEPEP_MIN_CUDA_ARCH_MAJOR}.0+)"
      continue
    fi
    normalized+="${normalized:+ }${stripped}"
  done
  [[ -n "${normalized}" ]] || die "CUDA arch list is empty after normalization (need SM ${DEEPEP_MIN_CUDA_ARCH_MAJOR}.0+)"
  if [[ "${raw}" != "${normalized}" ]]; then
    log "note: normalized CUDA arch list for DeepEP"
    log "  from: ${raw}"
    log "  to:   ${normalized}"
  fi
  printf '%s' "${normalized}"
}

cuda_arch_list_contains() {
  local list="${1}"
  local arch="${2}"
  local entry
  local IFS=' '
  for entry in ${list}; do
    if [[ "${entry}" == "${arch}" ]]; then
      return 0
    fi
  done
  return 1
}

assert_post_hopper_arch_list() {
  local list="${1}"
  local required
  for required in 9.0 10.0 12.0; do
    cuda_arch_list_contains "${list}" "${required}" || die "post-Hopper build must include SM ${required} (got: ${list})"
  done
}

normalize_wheel_profile() {
  case "${1}" in
    all) printf '%s' "all" ;;
    ampere | pre-hopper-sm80-sm89) printf '%s' "pre-hopper-sm80-sm89" ;;
    hopper | post-hopper-sm90-sm120 | post-hopper-sm90-sm100-sm120) printf '%s' "post-hopper-sm90-sm100-sm120" ;;
    *)
      die "unknown DEEPEP_WHEEL_PROFILE=${1} (use pre-hopper-sm80-sm89, post-hopper-sm90-sm100-sm120, all, or legacy ampere/hopper)"
      ;;
  esac
}

# PEP 440 local-version label (alphanumeric; no underscores). Profile → lustre wheel name.
wheel_local_version_label_for_profile() {
  case "${1}" in
    pre-hopper-sm80-sm89) printf '%s' "prehopper8089" ;;
    post-hopper-sm90-sm100-sm120) printf '%s' "posthopper90120" ;;
    *) die "no wheel local-version label for profile=${1}" ;;
  esac
}

# pip wheel → deep_ep-1.2.1+{commit}-cp312-...; tag as deep_ep-1.2.1+{commit}.{label}-cp312-...
tag_wheel_with_profile() {
  local built="${1}"
  local profile="${2}"
  local label wheel base dir tagged_base
  label="$(wheel_local_version_label_for_profile "${profile}")"
  if [[ "${built}" == *"+${label}-"* ]] || [[ "${built}" == *".${label}-"* ]]; then
    printf '%s' "${built}"
    return
  fi
  base="$(basename "${built}")"
  dir="$(dirname "${built}")"
  if [[ "${base}" =~ ^(deep_ep-.+\+[^./-]+)(-cp.*)$ ]]; then
    tagged_base="${BASH_REMATCH[1]}.${label}${BASH_REMATCH[2]}"
    wheel="${dir}/${tagged_base}"
  else
    die "unexpected wheel name from pip: ${built}"
  fi
  mv -f "${built}" "${wheel}"
  printf '%s' "${wheel}"
}

apply_profile() {
  local profile
  profile="$(normalize_wheel_profile "${1}")"
  case "${profile}" in
    pre-hopper-sm80-sm89)
      # Internode.cu uses Hopper-only PTX (elect, cp.async.bulk, cluster mbarrier). It cannot
      # be assembled for SM 8.x. Use DISABLE_SM90_FEATURES + no NVSHMEM → intranode only.
      _raw_arch="${DEEPEP_CUDA_ARCH_LIST:-${DEEPEP_PRE_HOPPER_CUDA_ARCH_LIST}}"
      export DISABLE_SM90_FEATURES=1
      export DISABLE_AGGRESSIVE_PTX_INSTRS=1
      export HYBRID_EP_MULTINODE=0
      DEEPEP_INSTALL_NVSHMEM=0
      DEEPEP_BUILD_RDMA=0
      DEEPEP_HIDE_NVSHMEM=1
      ;;
    post-hopper-sm90-sm100-sm120)
      _raw_arch="${DEEPEP_CUDA_ARCH_LIST:-${DEEPEP_POST_HOPPER_CUDA_ARCH_LIST}}"
      unset DISABLE_SM90_FEATURES
      export DISABLE_AGGRESSIVE_PTX_INSTRS=1
      export HYBRID_EP_MULTINODE=1
      DEEPEP_INSTALL_NVSHMEM=1
      DEEPEP_BUILD_RDMA=1
      DEEPEP_HIDE_NVSHMEM=0
      ;;
    *)
      die "unknown DEEPEP_WHEEL_PROFILE=${profile}"
      ;;
  esac

  if [[ -n "${DEEPEP_CUDA_ARCH_LIST:-}" ]]; then
    _raw_arch="${DEEPEP_CUDA_ARCH_LIST}"
  elif [[ "${DEEPEP_USE_ENV_TORCH_ARCH_LIST:-0}" == "1" && -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    _raw_arch="${TORCH_CUDA_ARCH_LIST}"
  fi

  export TORCH_CUDA_ARCH_LIST
  TORCH_CUDA_ARCH_LIST="$(normalize_cuda_arch_list "${_raw_arch}")"
  if [[ "${profile}" == "post-hopper-sm90-sm100-sm120" ]]; then
    assert_post_hopper_arch_list "${TORCH_CUDA_ARCH_LIST}"
  fi
  export TORCH_CUDA_ARCH_LIST
  export MAX_JOBS
  ACTIVE_PROFILE="${profile}"
}

preflight() {
  require_cmd git
  require_cmd nvcc
  require_cmd python3
  require_cmd pip

  [[ -f "${DEEPEP_PATCH}" ]] || die "patch not found: ${DEEPEP_PATCH}"
  mkdir -p "${OUTPUT_DIR}"

  if ! python3 -c "import torch" >/dev/null 2>&1; then
    die "PyTorch not importable; use the nemo-automodel container Python environment"
  fi

  log "profile           = ${ACTIVE_PROFILE}"
  log "project root      = ${PROJECT_ROOT}"
  log "output dir        = ${OUTPUT_DIR}"
  log "deepep commit     = ${DEEPEP_COMMIT}"
  log "cuda arch list    = ${TORCH_CUDA_ARCH_LIST}"
  log "DISABLE_SM90      = ${DISABLE_SM90_FEATURES:-0}"
  log "max jobs          = ${MAX_JOBS}"
  log "hybrid multinode  = ${HYBRID_EP_MULTINODE}"
  log "install nvshmem   = ${DEEPEP_INSTALL_NVSHMEM}"
  log "build root        = ${BUILD_ROOT}"
  nvcc --version | head -n 1
  python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
}

build_rdma_core() {
  log "building rdma-core ${RDMA_CORE_TAG}"
  git clone https://github.com/linux-rdma/rdma-core.git "${BUILD_ROOT}/rdma-core"
  (
    cd "${BUILD_ROOT}/rdma-core"
    git checkout "tags/${RDMA_CORE_TAG}"
    sh build.sh
  )
  export RDMA_CORE_HOME="${BUILD_ROOT}/rdma-core/build"
  log "RDMA_CORE_HOME=${RDMA_CORE_HOME}"
}

prepare_deepep_source() {
  if [[ -d "${BUILD_ROOT}/DeepEP/.git" ]]; then
    log "reusing DeepEP source tree"
    return
  fi
  log "cloning DeepEP (hybrid-ep) at ${DEEPEP_COMMIT}"
  git clone --branch hybrid-ep https://github.com/deepseek-ai/DeepEP.git "${BUILD_ROOT}/DeepEP"
  (
    cd "${BUILD_ROOT}/DeepEP"
    git fetch origin "${DEEPEP_COMMIT}"
    git checkout FETCH_HEAD
    patch -p1 < "${DEEPEP_PATCH}"
  )
}

hide_nvshmem_for_pre_hopper_build() {
  if [[ "${DEEPEP_HIDE_NVSHMEM:-0}" != "1" ]]; then
    return
  fi
  if python3 -c "import nvidia.nvshmem" >/dev/null 2>&1; then
    log "uninstalling NVSHMEM for pre-Hopper build (setup.py requires no NVSHMEM when DISABLE_SM90_FEATURES=1)"
    pip uninstall -y nvidia-nvshmem-cu13 2>/dev/null || true
  fi
}

install_build_deps() {
  if [[ "${DEEPEP_INSTALL_NVSHMEM:-0}" == "1" ]]; then
    log "installing NVSHMEM ${NVSHMEM_VERSION}"
    pip install --no-cache-dir "nvidia-nvshmem-cu13==${NVSHMEM_VERSION}"
  else
    hide_nvshmem_for_pre_hopper_build
  fi

  if [[ "${HYBRID_EP_MULTINODE}" == "1" ]]; then
    if command -v apt-get >/dev/null 2>&1; then
      log "installing libnvidia-ml-dev (multinode link; build-time only)"
      apt-get update
      apt-get install -y --no-install-recommends libnvidia-ml-dev
    else
      log "warning: apt-get unavailable; multinode build may fail without libnvidia-ml-dev"
    fi
  fi
}

build_wheel() {
  log "building ${ACTIVE_PROFILE} wheel (may take 15-90+ minutes depending on arch count)"
  mkdir -p "${OUTPUT_DIR}"
  (
    cd "${BUILD_ROOT}/DeepEP"
    pip wheel . --no-build-isolation -w "${OUTPUT_DIR}"
  )

  local built wheel
  built="$(ls -t "${OUTPUT_DIR}"/deep_ep-*.whl 2>/dev/null | head -1 || true)"
  [[ -n "${built}" ]] || die "no deep_ep-*.whl produced in ${OUTPUT_DIR}"

  wheel="$(tag_wheel_with_profile "${built}" "${ACTIVE_PROFILE}")"
  BUILT_WHEEL="${wheel}"
  log "wheel written: ${BUILT_WHEEL}"
  ls -lh "${BUILT_WHEEL}"

  # pip wheel may drop dependency wheels (e.g. nvidia_ml_py) into the same directory.
  find "${OUTPUT_DIR}" -maxdepth 1 -type f -name '*.whl' ! -name 'deep_ep-*.whl' -delete
}

append_manifest() {
  local manifest="${MANIFEST_FILE}"
  local wheel_rel="${BUILT_WHEEL}"
  if [[ "${BUILT_WHEEL}" == "${PROJECT_ROOT}/"* ]]; then
    wheel_rel="${BUILT_WHEEL#"${PROJECT_ROOT}/"}"
  elif [[ "${BUILT_WHEEL}" == "${OUTPUT_DIR}/"* ]]; then
    wheel_rel="wheels/deepep/$(basename "${BUILT_WHEEL}")"
  fi
  {
    echo "built_at=$(date -Iseconds)"
    echo "profile=${ACTIVE_PROFILE}"
    echo "wheel=${wheel_rel}"
    echo "deepep_commit=${DEEPEP_COMMIT}"
    echo "deepep_patch=${DEEPEP_PATCH}"
    echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
    echo "DISABLE_SM90_FEATURES=${DISABLE_SM90_FEATURES:-0}"
    echo "DISABLE_AGGRESSIVE_PTX_INSTRS=${DISABLE_AGGRESSIVE_PTX_INSTRS:-0}"
    echo "HYBRID_EP_MULTINODE=${HYBRID_EP_MULTINODE}"
    echo "NVSHMEM_VERSION=${NVSHMEM_VERSION}"
    echo "RDMA_CORE_TAG=${RDMA_CORE_TAG}"
    echo "MAX_JOBS=${MAX_JOBS}"
    python3 -c "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda}')"
    nvcc --version | head -n 1
    echo "install_command=pip uninstall -y deep_ep && pip install --force-reinstall ${wheel_rel}"
    echo "---"
  } >>"${manifest}"
}

run_profile_build() {
  local profile="${1}"
  apply_profile "${profile}"
  preflight

  if [[ "${DEEPEP_BUILD_RDMA:-0}" == "1" ]]; then
    build_rdma_core
  else
    unset RDMA_CORE_HOME || true
  fi

  prepare_deepep_source
  install_build_deps
  build_wheel
  append_manifest
}

main() {
  : >"${MANIFEST_FILE}"

  case "$(normalize_wheel_profile "${DEEPEP_WHEEL_PROFILE}")" in
    all)
      rm -rf "${BUILD_ROOT}"
      mkdir -p "${BUILD_ROOT}"
      run_profile_build pre-hopper-sm80-sm89
      run_profile_build post-hopper-sm90-sm100-sm120
      log "built both wheels; install the one matching your GPU (see manifest)"
      ;;
    pre-hopper-sm80-sm89 | post-hopper-sm90-sm100-sm120)
      local profile
      profile="$(normalize_wheel_profile "${DEEPEP_WHEEL_PROFILE}")"
      rm -rf "${BUILD_ROOT}"
      mkdir -p "${BUILD_ROOT}"
      run_profile_build "${profile}"
      ;;
  esac

  log "manifest written: ${MANIFEST_FILE}"
}

main "$@"
