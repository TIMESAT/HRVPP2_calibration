# --- START: squash to a single clean commit and push ---
set -euo pipefail

# 0) Run from repo root
git rev-parse --is-inside-work-tree >/dev/null

# 1) Ignore generated outputs going forward (adjust the patterns as needed)
printf "output/*.csv\n" >> .gitignore
git rm -r --cached output 2>/dev/null || true
git add .gitignore
git diff --cached --quiet || git commit -m "Ignore generated outputs"

# 2) Safety: tag current tip so you can recover old history locally if needed
git tag backup-pre-orphan-$(date +%Y%m%d-%H%M%S)

# 3) Create an orphan branch (no history) and commit current tree as root
# If your Git is older and lacks `switch`, use the 'checkout' variant below.
git switch --orphan temp-clean
git reset --hard
git add -A
git commit -m "Initial commit (history squashed)"

# 4) Replace 'main' with this orphan branch
git branch -M main

# 5) Force-push to GitHub (overwrites remote main)
git push --force --set-upstream origin main
# --- END ---
