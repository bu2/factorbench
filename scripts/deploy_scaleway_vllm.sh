#!/usr/bin/env bash

set -euo pipefail

# Minimal setup for vLLM on Ubuntu 24.04 (Noble) with NVIDIA L4.
# - Installs Docker and NVIDIA Container Toolkit
# - Runs vLLM OpenAI server in a GPU-enabled container
#
# Defaults:
#   Model: Qwen/Qwen3-4B-Thinking-2507-FP8
#   Port:  8000 (override with PORT env or --port)
#
# Optional env:
#   HF_TOKEN        Passes to container as HUGGING_FACE_HUB_TOKEN
#   VLLM_IMAGE      Override image (default: vllm/vllm-openai:latest-cuda12)
#   VLLM_ARGS       Extra args appended to vLLM server
#
# Usage examples:
#   sudo bash scripts/deploy_scaleway_vllm.sh
#   sudo bash scripts/deploy_scaleway_vllm.sh --model meta-llama/Llama-3.1-8B-Instruct
#   PORT=9000 HF_TOKEN=... sudo -E bash scripts/deploy_scaleway_vllm.sh -m mistralai/Mistral-7B-Instruct-v0.3

MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507-FP8}"
PORT="${PORT:-8000}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
EXTRA_ARGS="${VLLM_ARGS:-}"
NO_RUN=0

usage() {
  cat <<EOF
Deploy vLLM OpenAI server on Ubuntu Noble (GPU: NVIDIA L4)

Options:
  -m, --model MODEL   Hugging Face model id (default: ${MODEL})
  -p, --port PORT     Host port to expose (default: ${PORT})
      --image IMAGE   vLLM image (default: ${IMAGE})
      --no-run        Only install dependencies; do not start container
  -h, --help          Show this help

Env:
  HF_TOKEN     Passed to container as HUGGING_FACE_HUB_TOKEN
  VLLM_ARGS    Extra arguments appended to vLLM server
  VLLM_IMAGE   Override default image

Examples:
  sudo bash scripts/deploy_scaleway_vllm.sh
  sudo bash scripts/deploy_scaleway_vllm.sh -m meta-llama/Llama-3.1-8B-Instruct
  PORT=9000 HF_TOKEN=... sudo -E bash scripts/deploy_scaleway_vllm.sh \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 --image vllm/vllm-openai:latest-cuda12
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)
      [[ $# -ge 2 ]] || { echo "Missing value for $1"; exit 1; }
      MODEL="$2"; shift 2;;
    -p|--port)
      [[ $# -ge 2 ]] || { echo "Missing value for $1"; exit 1; }
      PORT="$2"; shift 2;;
    --image)
      [[ $# -ge 2 ]] || { echo "Missing value for $1"; exit 1; }
      IMAGE="$2"; shift 2;;
    --no-run)
      NO_RUN=1; shift;;
    -h|--help)
      usage; exit 0;;
    --)
      shift; break;;
    -*)
      echo "Unknown option: $1"; usage; exit 1;;
    *)
      # Positional model override
      MODEL="$1"; shift;;
  esac
done

log() { echo "[deploy] $*"; }

SUDO=""
if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then SUDO="sudo"; fi

log "Updating system packages..."
$SUDO apt-get update -y
$SUDO apt-get install -y python3.12-venv

# $SUDO apt-get install -y ca-certificates curl gnupg lsb-release
# log "Installing Docker Engine..."
# $SUDO install -m 0755 -d /etc/apt/keyrings
# if [[ ! -f /etc/apt/keyrings/docker.gpg ]]; then
#   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | $SUDO gpg --dearmor -o /etc/apt/keyrings/docker.gpg
#   $SUDO chmod a+r /etc/apt/keyrings/docker.gpg
# fi
# CODENAME=$( . /etc/os-release && echo "$UBUNTU_CODENAME" )
# echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${CODENAME} stable" | $SUDO tee /etc/apt/sources.list.d/docker.list > /dev/null
# $SUDO apt-get update -y
# $SUDO apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# log "Installing NVIDIA Container Toolkit..."
# distribution=$(. /etc/os-release; echo ${ID}${VERSION_ID})
# curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | $SUDO gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
# curl -fsSL https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list | \
#   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' | \
#   $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
# $SUDO apt-get update -y
# $SUDO apt-get install -y nvidia-container-toolkit

# log "Configuring Docker to use NVIDIA runtime..."
# $SUDO nvidia-ctk runtime configure --runtime=docker >/dev/null 2>&1 || true
# $SUDO systemctl enable --now docker
# $SUDO systemctl restart docker

# if ! command -v docker >/dev/null 2>&1; then
#   echo "Docker not found after installation." >&2
#   exit 1
# fi

if command -v nvidia-smi >/dev/null 2>&1; then
  log "Detected NVIDIA driver:"
  nvidia-smi || true
else
  log "Warning: nvidia-smi not found on host. Ensure NVIDIA drivers are installed."
fi

log "Testing Nvidia Container Engine..."
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi

log "Preparing persistent caches at /var/lib/vllm ..."
$SUDO mkdir -p /var/lib/vllm/hf-cache /var/lib/vllm/model-cache
OWNER_UID=${SUDO_UID:-$(id -u)}
OWNER_GID=${SUDO_GID:-$(id -g)}
$SUDO chown -R "$OWNER_UID:$OWNER_GID" /var/lib/vllm

log "Pulling image: ${IMAGE} ..."
sudo docker pull "${IMAGE}"

if [[ "$NO_RUN" -eq 1 ]]; then
  log "--no-run specified; skipping container start. Done."
  exit 0
fi

CONTAINER_NAME="vllm-server"
if sudo docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  log "Container ${CONTAINER_NAME} exists; replacing it..."
  sudo docker rm -f "${CONTAINER_NAME}" || true
fi

log "Installing Python dependencies..."
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
deactivate

log "Starting vLLM server on port ${PORT} with model: ${MODEL}"

RUN_ARGS=(
  "--name" "${CONTAINER_NAME}"
  "--restart" "unless-stopped"
  "-d"
  "--gpus" "all"
  "-p" "${PORT}:8000"
  "-v" "/var/lib/vllm/hf-cache:/root/.cache/huggingface"
  "-v" "/var/lib/vllm/model-cache:/models"
  "--shm-size" "2g"
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  RUN_ARGS+=("-e" "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}")
fi

sudo docker run "${RUN_ARGS[@]}" "${IMAGE}" \
  --model "${MODEL}" \
  --host 0.0.0.0 \
  --port 8000 \
  --download-dir /models \
  --trust-remote-code \
  ${EXTRA_ARGS}

log "vLLM is starting. Health check and usage:"
echo "  curl http://127.0.0.1:${PORT}/v1/models"
echo "  curl http://127.0.0.1:${PORT}/v1/chat/completions \
    -H 'Content-Type: application/json' -H 'Authorization: Bearer dev' \
    -d '{"model":"${MODEL}","messages":[{"role":"user","content":"Hello"}]}'"

sudo docker logs -f vllm-server

