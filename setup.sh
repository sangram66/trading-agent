#!/usr/bin/env bash
# One-shot setup + verification. Refuses to declare success unless every gate passes.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> creating virtualenv"
python3 -m venv .venv
source .venv/bin/activate

echo "==> installing dependencies"
pip install -q --upgrade pip
pip install -q -r requirements.txt

# `|| fail=1` is load-bearing: with `set -e` and `pipefail` a failing test suite
# would abort the script here, skipping the diagnosis below and leaving the user
# with a bare exit code and no idea which suite broke.
fail=0
echo "==> engine gates"
python3 -m tests.test_engine | tee    /tmp/ta_tests.log | tail -2 || fail=1
echo "==> multi-agent gates"
python3 -m tests.test_agent  | tee -a /tmp/ta_tests.log | tail -2 || fail=1

# Both suites must report "0 failed". Counting rather than a single grep:
# otherwise this passes when one suite succeeds and the other never ran.
clean=$(grep -c '0 failed' /tmp/ta_tests.log || true)

if [ "$fail" = "0" ] && [ "$clean" = "2" ]; then
  echo
  echo "SETUP OK — activate with:  source .venv/bin/activate"
  echo "Then run:                  python3 run_vol_lab.py"
else
  echo
  echo "SETUP FAILED — $clean/2 suites clean. Do not run research until green." >&2
  grep -E '^  \[FAIL\]|^FAILED:' /tmp/ta_tests.log >&2 || true
  exit 1
fi
