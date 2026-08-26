#!/usr/bin/env bash
# Deploy to GitHub Actions. Idempotent — safe to re-run.
#
# Everything this sets up is free: private repos get 2,000 Actions minutes a
# month and this uses roughly 120.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO_NAME="${1:-trading-agent}"

command -v gh >/dev/null || {
  echo "gh CLI not found. Install: https://cli.github.com" >&2; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "Run: gh auth login" >&2; exit 1; }

echo "==> dry run: replaying both workflows against a throwaway repo"
# Catches missing modules, gitignore mistakes and push races here, rather than
# at 22:30 UTC on a Tuesday with nobody watching.
./deploy/dry-run.sh >/dev/null 2>&1 || {
  echo "dry run failed — run ./deploy/dry-run.sh to see why" >&2; exit 1; }
echo "    workflows replay cleanly"

echo "==> verifying gates before deploying anything"
# Refuse to ship a broken engine. Once this runs on a schedule nobody is
# watching it, and a statistics bug would quietly poison the permanent record.
./setup.sh >/dev/null 2>&1 || { echo "gates failed — fix before deploying" >&2; exit 1; }
echo "    all gates green"

if [ ! -d .git ]; then
  echo "==> git init"
  git init -q -b main
  git add -A && git commit -qm "initial commit"
fi

if ! gh repo view "$REPO_NAME" >/dev/null 2>&1; then
  echo "==> creating PRIVATE repo $REPO_NAME"
  # Private matters twice: it keeps your research private, and the 60-day
  # scheduled-workflow auto-disable rule applies only to public repositories.
  gh repo create "$REPO_NAME" --private --source=. --remote=origin --push
else
  echo "==> repo exists, pushing"
  git remote get-url origin >/dev/null 2>&1 || \
    gh repo set-default "$REPO_NAME" >/dev/null 2>&1 || true
  git push -u origin main
fi

echo "==> secrets"
read -rp "    contact email for SEC User-Agent (required by sec.gov): " SEC_EMAIL
gh secret set SEC_CONTACT --body "$SEC_EMAIL"

read -rp "    alert webhook URL for failed runs (blank to skip): " HOOK
[ -n "$HOOK" ] && gh secret set ALERT_WEBHOOK --body "$HOOK" || \
  echo "    !! skipped — you will not be told when a scheduled run dies silently"

read -rp "    Anthropic API key for agent runs (blank to skip): " -s KEY; echo
[ -n "$KEY" ] && gh secret set ANTHROPIC_API_KEY --body "$KEY" || \
  echo "    skipped — ingest still works, agent-run will not"

echo
echo "==> triggering first ingest now (do not wait for the cron)"
gh workflow run ingest.yml || echo "    trigger it manually from the Actions tab"

cat <<'DONE'

Deployed.

  ingest      weekdays 22:30 UTC   ~2 min    free
  agent-run   Saturdays 03:00 UTC  on demand free compute, you pay tokens

Both share the `repo-writer` concurrency group, so the trial ledger has exactly
one writer at a time.

Check it worked:   gh run list --limit 5
Watch the first:   gh run watch
DONE
