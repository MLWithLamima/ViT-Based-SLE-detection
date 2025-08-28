#!/usr/bin/env bash
# Initialize repo with LFS and push to GitHub (macOS/Linux)
# Usage: GITHUB_USER=<you> REPO=<repo> ./scripts/init_repo.sh

set -euo pipefail
GITHUB_USER="${GITHUB_USER:-<your-username>}"
REPO="${REPO:-<your-repo>}"

git init
git lfs install
git add .
git commit -m "Initial commit: thesis project"
git branch -M main
git remote add origin git@github.com:${GITHUB_USER}/${REPO}.git
git push -u origin main