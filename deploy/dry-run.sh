#!/usr/bin/env bash
# Replay both GitHub Actions workflows locally against a throwaway bare repo.
#
# Catches the failures that would otherwise show up at 22:30 UTC on a Tuesday:
# a missing module, a gitignore that swallows the data, a push that races.
# Everything except GitHub's runner itself.
#
#   ./deploy/dry-run.sh
set -uo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
fail=0
step() { printf "\n\033[1m== %s\033[0m\n" "$1"; }
ok()   { echo "   PASS  $1"; }
bad()  { echo "   FAIL  $1"; fail=1; }

step "0. every module the workflows call must exist"
for m in engine.ingest.collect agent.run tests.test_engine tests.test_agent; do
  f="${m//.//}.py"
  [ -f "$f" ] && ok "$m" || bad "$m missing (referenced by a workflow)"
done

step "1. workflow YAML parses and shares one writer queue"
python3 - <<'PY' || fail=1
import sys, yaml
groups = {}
for f in ("ingest", "agent-run"):
    d = yaml.safe_load(open(f".github/workflows/{f}.yml"))
    groups[f] = d["concurrency"]["group"]
    assert d["concurrency"]["cancel-in-progress"] is False, f"{f}: must not cancel mid-write"
    print(f"   PASS  {f}.yml parses, group={groups[f]}")
assert len(set(groups.values())) == 1, f"writer queues differ: {groups} — the ledger can race"
print("   PASS  both workflows serialise through one queue")
PY

step "2. fake origin + clone"
git init -q --bare -b main "$WORK/origin.git"
# Copy with tar rather than rsync: rsync is not installed everywhere, and when
# it was missing the `cd` below silently failed and the whole dry run proceeded
# to operate on the real repository. Never let a harness fall through to the
# source tree.
mkdir -p "$WORK/checkout"
tar -C "$REPO_ROOT" --exclude=./.git --exclude=./.venv --exclude=./data \
    --exclude=__pycache__ -cf - . | tar -C "$WORK/checkout" -xf -
cd "$WORK/checkout" || { echo "FATAL: could not enter sandbox checkout" >&2; exit 1; }
git init -q -b main && git config user.email ci@local && git config user.name ci
git add -A && git commit -qm "initial" && git remote add origin "$WORK/origin.git" && git push -q -u origin main
ok "pushed $(git ls-files | wc -l | tr -d ' ') tracked files"

step "3. replay ingest.yml"
python3 -m venv .venv >/dev/null 2>&1
./.venv/bin/pip install -q -r requirements.txt 2>&1 | tail -2
./.venv/bin/python3 -m engine.ingest.collect 2>/dev/null | tail -2
./.venv/bin/python3 -m engine.ingest.collect --verify-only && ok "manifest verified" || bad "manifest verify"
git add data/
git diff --staged --quiet && bad "nothing staged — is data/ gitignored?" || ok "data staged"
[ "$(git diff --staged --name-only | grep -c '\.cache')" = "0" ] \
  && ok "http cache correctly excluded" || bad "cache is being committed"
git commit -qm "data: $(date -u +%Y-%m-%d)"
git push -q && ok "ingest pushed" || bad "ingest push"

step "4. replay agent-run.yml (with a racing ingest commit)"
# Simulate the ingest job landing a commit while the agent run was working.
git clone -q "$WORK/origin.git" "$WORK/other" && cd "$WORK/other" || exit 1
git config user.email x@local && git config user.name x
echo late > data/racing.txt && git add -A && git commit -qm "data: racing" && git push -q
cd "$WORK/checkout" || exit 1
./.venv/bin/python3 -m agent.run --max-trials 2 --agents athena --n-sim 100 2>&1 | tail -3
git add research/ data/
git diff --staged --quiet || git commit -qm "research: agent run"
git push -q 2>/dev/null && bad "stale push should have been rejected" || ok "stale push rejected (expected)"
pushed=0
for i in 1 2 3; do
  git pull --rebase --autostash -q && git push -q && { ok "rebase-retry pushed on attempt $i"; pushed=1; break; }
  sleep 1
done
[ "$pushed" = "1" ] || bad "could not push after 3 attempts"

step "5. both writers' work survived"
git clone -q "$WORK/origin.git" "$WORK/final" && cd "$WORK/final" || exit 1
[ -f data/racing.txt ] && ok "racing ingest commit preserved" || bad "ingest commit lost"
ls research/findings/*.json >/dev/null 2>&1 && ok "agent findings preserved" || bad "agent findings lost"
find data -name '*.parquet' | head -3 | sed 's/^/         /'
echo "         repo size: $(du -sh .git | cut -f1)"

printf "\n%s\n" "----------------------------------------------------------"
[ "$fail" = "0" ] && echo "DRY RUN GREEN — safe to run ./deploy/bootstrap.sh" \
                  || echo "DRY RUN FAILED — fix before deploying"
exit $fail
