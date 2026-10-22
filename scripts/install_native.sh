#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
if [[ ! -d third_party/eigen/Eigen ]]; then
  git clone --depth 1 https://gitlab.com/libeigen/eigen.git third_party/eigen
fi
if [[ ! -d third_party/glm/glm ]]; then
  git clone --depth 1 https://github.com/g-truc/glm.git third_party/glm
fi
if [[ ! -d third_party/simple-knn ]]; then
  git clone --depth 1 https://gitlab.inria.fr/bkerbl/simple-knn.git third_party/simple-knn
fi
python -m pip install -e third_party/simple-knn
python -m pip install --no-build-isolation -e third_party/diff-gaussian-rasterization
python -m pip install --no-build-isolation -e .
