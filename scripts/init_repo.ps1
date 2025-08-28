# Initialize repo with LFS and push to GitHub (Windows PowerShell)
# Usage: powershell -ExecutionPolicy Bypass -File scripts\init_repo.ps1 -GitHubUser "<you>" -Repo "<repo>"

param(
  [string]$GitHubUser = "<your-username>",
  [string]$Repo = "<your-repo>"
)

git init
git lfs install
git add .
git commit -m "Initial commit: thesis project"
git branch -M main
git remote add origin ("git@github.com:{0}/{1}.git" -f $GitHubUser, $Repo)
git push -u origin main