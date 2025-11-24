# Setting Up GitHub Repository

Your local git repository is ready! Follow these steps to create the GitHub repo and push your code.

## Option 1: Using GitHub CLI (Recommended)

If you have `gh` installed:

```bash
gh repo create efficiency_of_reason --public --source=. --remote=origin --push
```

## Option 2: Using GitHub Web Interface

1. **Create the repository on GitHub:**
   - Go to https://github.com/new
   - Repository name: `efficiency_of_reason`
   - Description: "Quantifying the Computational Sparsity of Chain-of-Thought in Large Language Models"
   - Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
   - Click "Create repository"

2. **Push your code:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/efficiency_of_reason.git
   git push -u origin main
   ```

   Replace `YOUR_USERNAME` with your GitHub username.

## Option 3: Using SSH

If you prefer SSH:

```bash
git remote add origin git@github.com:YOUR_USERNAME/efficiency_of_reason.git
git push -u origin main
```

## For RunPod

Once the repo is on GitHub, you can clone it on RunPod:

```bash
git clone https://github.com/YOUR_USERNAME/efficiency_of_reason.git
cd efficiency_of_reason
pip install -r requirements.txt
python scripts/01_run_sparsity_gap.py
```

## Current Status

✅ Git repository initialized
✅ All files committed
✅ Branch renamed to `main`
✅ Ready to push to GitHub

