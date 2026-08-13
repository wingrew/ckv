#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"
cp "${ROOT_DIR}/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json" /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/
mv "${ROOT_DIR}/openai_dataset.py" /sgl-workspace/sglang/python/sglang/benchmark/datasets/openai_dataset.py
mv "${ROOT_DIR}/schedule_policy.py" /sgl-workspace/sglang/python/sglang/srt/managers/schedule_policy.py
mv "${ROOT_DIR}/scheduler.py" /sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py
mv "${ROOT_DIR}/server_args.py" /sgl-workspace/sglang/python/sglang/srt/server_args.py
mv "${ROOT_DIR}/serving.py" /sgl-workspace/sglang/python/sglang/benchmark/serving.py
: "${MASTER_ADDR:?MASTER_ADDR is required}"
: "${WORLD_SIZE:?WORLD_SIZE is required}"
: "${RANK:?RANK is required}"
: "${METACAMP_LLM_PORT:?METACAMP_LLM_PORT is required}"

case "${WORLD_SIZE}" in
  1|2) ;;
  *) echo "WORLD_SIZE must be 1 or 2, got ${WORLD_SIZE}" >&2; exit 2 ;;
esac

if ! [[ "${RANK}" =~ ^[0-9]+$ ]] || (( RANK < 0 || RANK >= WORLD_SIZE )); then
  echo "RANK must be in [0, WORLD_SIZE), got ${RANK}" >&2
  exit 2
fi

if ! [[ "${METACAMP_LLM_PORT}" =~ ^[0-9]+$ ]] || \
   (( METACAMP_LLM_PORT < 1 || METACAMP_LLM_PORT > 65535 )); then
  echo "METACAMP_LLM_PORT must be a valid TCP port" >&2
  exit 2
fi

MODEL_PATH="${MODEL_PATH:-/metacamp/GLM-5.2-FP8}"
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
echo "===== MODEL DEBUG =====" >&2
echo "MODEL_PATH=${MODEL_PATH}" >&2

env | sort | grep -Ei \
  'MODEL|HF_|HUGGING|METACAMP|PUBLIC_DATA' >&2 || true

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2

  echo "===== /mnt =====" >&2
  ls -lah /mnt >&2 || true

  echo "===== /mnt/public_data =====" >&2
  ls -lah /mnt/public_data >&2 || true

  echo "===== config.json candidates =====" >&2
  find /mnt \
    -maxdepth 5 \
    -type f \
    -name config.json \
    -print 2>/dev/null | head -200 >&2 || true

  exit 1
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: MODEL_PATH exists but config.json is missing." >&2
  echo "MODEL_PATH=${MODEL_PATH}" >&2
  ls -lah "${MODEL_PATH}" >&2 || true
  exit 1
fi

echo "MODEL_PATH OK: ${MODEL_PATH}" >&2
echo "===== END MODEL DEBUG =====" >&2

MODEL_NAME="${MODEL_NAME:-GLM-5.2}"
BACKEND_LIMIT="${BACKEND_LIMIT:-12}"
BACKEND_PORT="${SGLANG_BACKEND_PORT:-31057}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"

# rank 0 must reserve METACAMP_LLM_PORT for the public gateway.
if (( RANK == 0 )) && (( BACKEND_PORT == METACAMP_LLM_PORT )); then
  BACKEND_PORT=31058
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1
export PREFILL_WARMUP_TOKENS=32768
export PREFILL_WARMUP_REQUESTS=1

for command in python3 sglang; do
  command -v "${command}" >/dev/null 2>&1 || {
    echo "missing required command: ${command}" >&2
    exit 127
  }
done

python3 - <<'PY'
import fastapi, httpx, uvicorn  # noqa: F401
PY

# SGLang already maps max_tokens/min_tokens/ignore_eos. This idempotent patch
# additionally applies min_tokens to string/regex/grammar stop conditions.
python3 "${ROOT_DIR}/patch_sglang_min_tokens.py"

REGISTRATION_TOKEN="$(
  MASTER_ADDR="${MASTER_ADDR}" \
  METACAMP_LLM_PORT="${METACAMP_LLM_PORT}" \
  WORLD_SIZE="${WORLD_SIZE}" \
  python3 - <<'PY'
import hashlib
import os
raw = f"{os.environ['MASTER_ADDR']}:{os.environ['METACAMP_LLM_PORT']}:{os.environ['WORLD_SIZE']}"
print(hashlib.sha256(raw.encode()).hexdigest())
PY
)"

SGLANG_PID=""
GATEWAY_PID=""

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  for pid in "${GATEWAY_PID:-}" "${SGLANG_PID:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  sleep 1
  for pid in "${GATEWAY_PID:-}" "${SGLANG_PID:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  exit "${code}"
}
trap cleanup EXIT INT TERM

http_ok() {
  local url=$1
  python3 - "${url}" <<'PY'
import sys
import urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
        raise SystemExit(0 if response.status < 500 else 1)
except Exception:
    raise SystemExit(1)
PY
}

wait_for_backend() {
  local url=$1
  local pid=$2
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  local log_file="${LOG_DIR}/sglang_rank${RANK}.log"

  until http_ok "${url}/health"; do
    # SGLang process exited before becoming ready
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "SGLang exited during startup." >&2
      echo "===== SGLANG RANK ${RANK} LOG BEGIN =====" >&2

      if [[ -f "${log_file}" ]]; then
        tail -n 1000 "${log_file}" >&2
      else
        echo "SGLang log file not found: ${log_file}" >&2
      fi

      echo "===== SGLANG RANK ${RANK} LOG END =====" >&2

      wait "${pid}" || true
      return 1
    fi

    # Startup timeout
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for ${url}/health" >&2
      echo "===== SGLANG RANK ${RANK} LOG BEGIN =====" >&2

      if [[ -f "${log_file}" ]]; then
        tail -n 1000 "${log_file}" >&2
      else
        echo "SGLang log file not found: ${log_file}" >&2
      fi

      echo "===== SGLANG RANK ${RANK} LOG END =====" >&2

      return 1
    fi

    sleep 1
  done

  echo "SGLang rank=${RANK} is ready: ${url}" >&2
}

url_host() {
  local host=$1
  if [[ "${host}" == *:* && "${host}" != \[*\] ]]; then
    printf '[%s]' "${host}"
  else
    printf '%s' "${host}"
  fi
}

resolve_local_ip() {
  MASTER_ADDR="${MASTER_ADDR}" python3 - <<'PY'
import os
import socket
master = os.environ["MASTER_ADDR"]
try:
    infos = socket.getaddrinfo(master, 9, type=socket.SOCK_DGRAM)
    for family, socktype, proto, _, sockaddr in infos:
        sock = socket.socket(family, socktype, proto)
        try:
            sock.connect(sockaddr)
            print(sock.getsockname()[0])
            raise SystemExit(0)
        finally:
            sock.close()
except Exception:
    pass
for family in (socket.AF_INET, socket.AF_INET6):
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, family):
            address = item[4][0]
            if not address.startswith("127.") and address != "::1":
                print(address)
                raise SystemExit(0)
    except Exception:
        pass
raise SystemExit("cannot determine a routable local IP")
PY
}

start_sglang() {
  local log_file="${LOG_DIR}/sglang_rank${RANK}.log"
  echo "starting SGLang rank=${RANK} backend_port=${BACKEND_PORT}" | tee -a "${log_file}"
  # sglang serve \
  #   --model-path "${MODEL_PATH}" \
  #   --served-model-name "${MODEL_NAME}" \
  #   --tp 8 \
  #   --host 0.0.0.0 \
  #   --port "${BACKEND_PORT}" \
  #   --trust-remote-code \
  #   --mem-fraction-static 0.81 \
  #   --max-running-requests "${BACKEND_LIMIT}" \
  #   --chunked-prefill-size 32768 \
  #   --max-prefill-tokens 65536 \
  #   --disable-radix-cache \
  #   --cuda-graph-backend-prefill breakable \
  #   --decode-log-interval 200 \
  #   --tool-call-parser glm47 \
  #   --watchdog-timeout 7200 \
  export SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384
  export SGLANG_FLASHINFER_ALLREDUCE_MAX_TOKENS=2048
  export SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP=1

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  sglang serve \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --tp 8 \
    --host 0.0.0.0 \
    --port "${BACKEND_PORT}" \
    --trust-remote-code \
    --mem-fraction-static 0.86 \
    --max-running-requests 16 \
    --chunked-prefill-size 32768 \
    --max-prefill-tokens 32768 \
    --schedule-policy sjf \
    --sjf-multi-request-prefill-tokens 16384 \
    --sjf-multi-request-prefill-threshold 2 \
    --sjf-multi-request-prefill-max-requests 2 \
    --sjf-multi-request-prefill-max-running-requests 10 \
    --tokenizer-worker-num 12 \
    --kv-cache-dtype fp8_e4m3 \
    --dsa-prefill-backend flashmla_sparse_q8 \
    --dsa-decode-backend flashmla_kv \
    --enable-prefill-cp \
    --attn-cp-size 8 \
    --cp-strategy interleave \
    --moe-dp-size 1 \
    --moe-a2a-backend none \
    --moe-runner-backend triton \
    --enable-fused-moe-sum-all-reduce \
    --enable-nccl-nvls \
    --tool-call-parser glm47 \
    --enable-metrics \
    --log-level info \
    --decode-log-interval 100 \
    --watchdog-timeout 7200 \
    &
  SGLANG_PID=$!
}
    # --speculative-algorithm EAGLE \
    # --speculative-num-steps 3 \
    # --speculative-eagle-topk 1 \
    # --speculative-num-draft-tokens 4 \
    #   --attn-cp-size 8 \
    # --enable-prefill-cp \
    # --cp-strategy interleave \
    # --enable-nccl-nvls \
    #   --dp 8 \
    # --enable-dp-attention \
register_rank1() {
  local backend_url=$1
  local master_url_host
  master_url_host="$(url_host "${MASTER_ADDR}")"
  local register_url="http://${master_url_host}:${METACAMP_LLM_PORT}/_internal/register"
  local deadline=$((SECONDS + STARTUP_TIMEOUT))

  while true; do
    if ! kill -0 "${SGLANG_PID}" 2>/dev/null; then
      echo "rank1 SGLang exited before registration" >&2
      return 1
    fi

    if REGISTER_URL="${register_url}" \
       BACKEND_URL="${backend_url}" \
       REGISTRATION_TOKEN="${REGISTRATION_TOKEN}" \
       python3 - <<'PY'
import json
import os
import urllib.request
request = urllib.request.Request(
    os.environ["REGISTER_URL"],
    data=json.dumps({"url": os.environ["BACKEND_URL"]}).encode(),
    headers={
        "content-type": "application/json",
        "x-metacamp-registration-token": os.environ["REGISTRATION_TOKEN"],
    },
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=3) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
    then
      echo "registered rank1 backend ${backend_url}"
      return 0
    fi

    if (( SECONDS >= deadline )); then
      echo "timed out registering rank1 at ${register_url}" >&2
      return 1
    fi
    sleep 1
  done
}

start_sglang
LOCAL_BACKEND_URL="http://127.0.0.1:${BACKEND_PORT}"
wait_for_backend "${LOCAL_BACKEND_URL}" "${SGLANG_PID}"

if (( PREFILL_WARMUP_TOKENS > 0 && PREFILL_WARMUP_REQUESTS > 0 )); then
  echo "Warming long-prefill kernels with ${PREFILL_WARMUP_REQUESTS} x ${PREFILL_WARMUP_TOKENS} synthetic tokens"
  python3 - "${LOCAL_BACKEND_URL}" "${PREFILL_WARMUP_TOKENS}" "${PREFILL_WARMUP_REQUESTS}" <<'PY'
import concurrent.futures
import json
import sys
import urllib.request

server_url = sys.argv[1]
token_count = int(sys.argv[2])
request_count = int(sys.argv[3])
body = json.dumps(
    {
        "input_ids": [42] * token_count,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
    }
).encode("ascii")


def warm_one(_):
    request = urllib.request.Request(
        f"{server_url}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        response.read()


with concurrent.futures.ThreadPoolExecutor(max_workers=request_count) as executor:
    list(executor.map(warm_one, range(request_count)))

flush = urllib.request.Request(f"{server_url}/flush_cache", data=b"")
with urllib.request.urlopen(flush, timeout=60) as response:
    response.read()
PY
fi

if (( RANK == 0 )); then
  sleep 60
  echo "SGLang rank=${RANK} is ready"
fi
if (( RANK == 1 )); then
  LOCAL_IP="$(resolve_local_ip)"
  LOCAL_URL_HOST="$(url_host "${LOCAL_IP}")"
  register_rank1 "http://${LOCAL_URL_HOST}:${BACKEND_PORT}"
  set +e
  wait "${SGLANG_PID}"
  STATUS=$?
  set -e
  exit "${STATUS}"
fi

python3 "${ROOT_DIR}/gateway.py" \
  --host 0.0.0.0 \
  --port "${METACAMP_LLM_PORT}" \
  --backend "${LOCAL_BACKEND_URL}" \
  --backend-limit "${BACKEND_LIMIT}" \
  --expected-backends "${WORLD_SIZE}" \
  --registration-token "${REGISTRATION_TOKEN}" \
  >"${LOG_DIR}/gateway_rank0.log" 2>&1 &
GATEWAY_PID=$!

echo "gateway listening on 0.0.0.0:${METACAMP_LLM_PORT}"

# Keep start.sh alive and fail the job if either rank-0 process exits.
while true; do
  if ! kill -0 "${SGLANG_PID}" 2>/dev/null; then
    echo "rank0 SGLang exited; see ${LOG_DIR}/sglang_rank0.log" >&2
    set +e; wait "${SGLANG_PID}"; STATUS=$?; set -e
    exit "${STATUS}"
  fi
  if ! kill -0 "${GATEWAY_PID}" 2>/dev/null; then
    echo "gateway exited; see ${LOG_DIR}/gateway_rank0.log" >&2
    set +e; wait "${GATEWAY_PID}"; STATUS=$?; set -e
    exit "${STATUS}"
  fi
  sleep 2
done
