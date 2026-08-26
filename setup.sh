#!/usr/bin/env bash
# One-shot setup + verification. Refuses to declare success unless every gate passes.
set -euo pipefail
cd "$(dirname "$0")"

# Pick an interpreter. `python3` is not reliable: on macOS it often resolves to
# Apple's stub in /usr/bin (3.9.x) even when a newer build sits in
# /usr/local/bin, and a venv built on the wrong one fails later in confusing
# ways. Override explicitly with:  PYTHON=/usr/local/bin/python3.12 ./setup.sh
pick_python() {
  if [ -n "${PYTHON:-}" ]; then echo "$PYTHON"; return; fi
  for c in python3.13 python3.12 python3.11 python3.10 python3; do
    p=$(command -v "$c" 2>/dev/null) || continue
    "$p" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)' \
      2>/dev/null && { echo "$p"; return; }
  done
  echo ""
}

PY_BIN="$(pick_python)"
if [ -z "$PY_BIN" ]; then
  echo "No Python 3.10+ found." >&2
  echo "Install one, or point at yours:  PYTHON=/usr/local/bin/python3.12 ./setup.sh" >&2
  exit 1
fi

PY_VER="$("$PY_BIN" -c 'import sys;print(".".join(map(str,sys.version_info[:3])))')"
echo "==> using $PY_BIN (Python $PY_VER)"

echo "==> creating virtualenv"
rm -rf .venv
"$PY_BIN" -m venv .venv
source .venv/bin/activate

# The venv must actually be the interpreter we chose. A stale .venv from a
# previous run with a different Python is a classic source of "it worked
# yesterday".
ACTUAL="$(python3 -c 'import sys;print(".".join(map(str,sys.version_info[:3])))')"
[ "$ACTUAL" = "$PY_VER" ] || {
  echo "venv reports Python $ACTUAL but $PY_BIN is $PY_VER — aborting" >&2; exit 1; }

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
