#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 OWNER/REPO"
  echo "Example: $0 username/ldwt-ir-taslp"
  exit 2
fi

REPO="$1"

if ! command -v git >/dev/null 2>&1; then
  echo "git is not installed or not on PATH."
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI gh is not installed or not on PATH."
  echo "Install gh or create the GitHub repository manually and push with git."
  exit 1
fi

if [[ ! -d .git ]]; then
  git init
  git branch -M main
fi

git add .
git commit -m "Initial reproducibility release" || true

if ! gh repo view "$REPO" >/dev/null 2>&1; then
  gh repo create "$REPO" --private --source=. --remote=origin --push
else
  git remote remove origin >/dev/null 2>&1 || true
  git remote add origin "https://github.com/${REPO}.git"
  git push -u origin main
fi
