#!/usr/bin/env bash
set -eu

code_root="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="${code_root}${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m unidrivevla.serve "$@"
