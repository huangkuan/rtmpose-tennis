#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RTMPOSE_PYTHON="${RTMPOSE_PYTHON:-python3.11}"
RTMPOSE_VENV="${RTMPOSE_VENV:-${PROJECT_DIR}/.venv311}"

if ! command -v "${RTMPOSE_PYTHON}" >/dev/null 2>&1; then
  echo "Python 3.11 was not found. On macOS with Homebrew, run:" >&2
  echo "  brew install python@3.11" >&2
  echo "Then rerun this script, or set RTMPOSE_PYTHON to its path." >&2
  exit 1
fi

PYTHON_VERSION="$("${RTMPOSE_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYTHON_VERSION}" != "3.11" ]]; then
  echo "Expected Python 3.11, found ${PYTHON_VERSION} at ${RTMPOSE_PYTHON}." >&2
  exit 1
fi

"${RTMPOSE_PYTHON}" -m venv "${RTMPOSE_VENV}"
VENV_PYTHON="${RTMPOSE_VENV}/bin/python"
VENV_MIM="${RTMPOSE_VENV}/bin/mim"

"${VENV_PYTHON}" -m pip install --upgrade pip wheel
"${VENV_PYTHON}" -m pip install "numpy>=1.24,<2" "torch==2.1.2" "torchvision==0.16.2"
"${VENV_PYTHON}" -m pip install "openmim==0.3.9"

# chumpy's legacy setup imports pip from its build environment. Disabling build
# isolation is required with modern pip and is safe inside this dedicated venv.
"${VENV_PYTHON}" -m pip install "setuptools<81"
"${VENV_PYTHON}" -m pip install --no-build-isolation "chumpy==0.70"

"${VENV_MIM}" install "mmengine>=0.8,<1"
"${VENV_MIM}" install "mmcv>=2.0.1,<2.2.0"
"${VENV_PYTHON}" -m pip install -e "${PROJECT_DIR}"

"${VENV_PYTHON}" -c \
  'import torch, mmcv, mmengine, mmdet, mmpose; print(f"Ready: torch={torch.__version__}, mmcv={mmcv.__version__}, mmpose={mmpose.__version__}")'

echo "Activate the environment with:"
echo "  source ${RTMPOSE_VENV}/bin/activate"
