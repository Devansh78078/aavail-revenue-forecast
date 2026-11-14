#!/bin/sh
set -eu
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec python -m unittest discover -s tests -v
