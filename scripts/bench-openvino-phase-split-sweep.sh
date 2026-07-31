#!/usr/bin/env bash
# Sweep -pg (prefill,decode) token counts for hybrid vs single-device on OpenVINO.
# Usage (on host with setupvars sourced):
#   BENCH=.../llama-bench ./scripts/bench-openvino-phase-split-sweep.sh model.gguf [pp,tg ...]
set -eo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 model.gguf [pp,tg ...]" >&2
  exit 1
fi

MODEL="$1"
shift
TAG="$(basename "$MODEL" .gguf)"
BENCH="${BENCH:-$HOME/src/llama/build_oneapi_master_openvino_RelWithDebInfo/bin/llama-bench}"
CACHE_ROOT="${CACHE_ROOT:-/tmp/ov_pg_sweep}"
REPS="${REPS:-2}"

if [[ ! -x "$BENCH" ]]; then
  echo "error: llama-bench not found: $BENCH" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  set -- \
    8,512 16,512 32,512 64,512 \
    8,1024 16,1024 32,1024 64,1024 128,1024 \
    512,8 512,16 512,32 512,64 \
    1024,8 1024,16 1024,32 1024,64 \
    2048,8 2048,16 2048,32 2048,64 \
    512,128 1024,128
fi

bench_one() {
  local pg="$1" cfg="$2"
  shift 2
  export GGML_OPENVINO_CACHE_DIR="${CACHE_ROOT}/${TAG}/${pg}/${cfg}"
  mkdir -p "$GGML_OPENVINO_CACHE_DIR"
  unset GGML_OPENVINO_PHASE_SPLIT GGML_OPENVINO_PREFILL_DEVICE GGML_OPENVINO_DECODE_DEVICE 2>/dev/null || true
  # shellcheck disable=SC2086
  line=$(env "$@" "$BENCH" -m "$MODEL" -r "$REPS" -o jsonl -pg "$pg" 2>/dev/null | tail -1)
  python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d['avg_ts'])" "$line"
}

echo "model,pg,config,avg_ts"
for pg in "$@"; do
  pp="${pg%,*}"
  tg="${pg#*,}"
  echo "pg ${pg} (pp=${pp} tg=${tg}) ..." >&2

  ts=$(bench_one "$pg" all_cpu GGML_OPENVINO_DEVICE=CPU GGML_OPENVINO_STATEFUL_EXECUTION=0)
  echo "${TAG},${pg},all_cpu,${ts}"

  ts=$(bench_one "$pg" all_igpu GGML_OPENVINO_DEVICE=GPU.0 GGML_OPENVINO_STATEFUL_EXECUTION=1)
  echo "${TAG},${pg},all_igpu,${ts}"

  ts=$(bench_one "$pg" split_cpu_pp_igpu_tg \
    GGML_OPENVINO_PHASE_SPLIT=1 GGML_OPENVINO_PREFILL_DEVICE=CPU GGML_OPENVINO_DECODE_DEVICE=GPU.0 \
    GGML_OPENVINO_STATEFUL_EXECUTION=1)
  echo "${TAG},${pg},split_cpu_pp_igpu_tg,${ts}"

  ts=$(bench_one "$pg" split_igpu_pp_cpu_tg \
    GGML_OPENVINO_PHASE_SPLIT=1 GGML_OPENVINO_PREFILL_DEVICE=GPU.0 GGML_OPENVINO_DECODE_DEVICE=CPU \
    GGML_OPENVINO_STATEFUL_EXECUTION=1)
  echo "${TAG},${pg},split_igpu_pp_cpu_tg,${ts}"
done
