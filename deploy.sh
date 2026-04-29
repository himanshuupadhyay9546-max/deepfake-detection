#!/bin/bash

# ═══════════════════════════════════════════════════════════════
#   DeepShield — Full Setup & Railway Deployment Script
#   Run this from inside your deepfake-detection project folder
# ═══════════════════════════════════════════════════════════════

set -e  # Stop on any error

# ── Colors ──────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# ── Helpers ─────────────────────────────────────────────────────
info()    { echo -e "${CYAN}[INFO]${RESET}  $1"; }
success() { echo -e "${GREEN}[OK]${RESET}    $1"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $1"; }
error()   { echo -e "${RED}[ERROR]${RESET} $1"; exit 1; }
step()    { echo -e "\n${BOLD}━━━  $1  ━━━${RESET}"; }

# ── Banner ───────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}${BOLD}"
echo "  ██████╗ ███████╗███████╗██████╗ ███████╗██╗  ██╗██╗███████╗██╗     ██████╗ "
echo "  ██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝██║  ██║██║██╔════╝██║     ██╔══██╗"
echo "  ██║  ██║█████╗  █████╗  ██████╔╝███████╗███████║██║█████╗  ██║     ██║  ██║"
echo "  ██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ╚════██║██╔══██║██║██╔══╝  ██║     ██║  ██║"
echo "  ██████╔╝███████╗███████╗██║     ███████║██║  ██║██║███████╗███████╗██████╔╝"
echo "  ╚═════╝ ╚══════╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝╚═════╝ "
echo -e "${RESET}"
echo -e "  ${BOLD}Deepfake Detection System — Auto Deploy Script${RESET}"
echo -e "  ${CYAN}Automates: venv → packages → config files → git → Railway${RESET}"
echo ""

# ── Check we're in the right folder ─────────────────────────────
step "Checking project folder"
if [ ! -f "main.py" ]; then
  error "main.py not found. Make sure you're inside the deepfake-detection folder.\n  Run: cd path/to/deepfake-detection"
fi
success "Found main.py — correct folder"

# ── Check Python ─────────────────────────────────────────────────
step "Checking Python installation"
if ! command -v python &>/dev/null && ! command -v python3 &>/dev/null; then
  error "Python not found. Install from https://python.org and add to PATH."
fi

PYTHON=$(command -v python3 || command -v python)
PY_VERSION=$($PYTHON --version 2>&1)
success "Found $PY_VERSION at $PYTHON"

# ── Virtual environment ──────────────────────────────────────────
step "Setting up virtual environment"
if [ -d "venv" ]; then
  warn "venv already exists — skipping creation"
else
  info "Creating virtual environment..."
  $PYTHON -m venv venv
  success "Virtual environment created"
fi

# Activate venv
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
  ACTIVATE="venv/Scripts/activate"
else
  ACTIVATE="venv/bin/activate"
fi

source "$ACTIVATE" 2>/dev/null || {
  warn "Could not auto-activate venv. Using system Python with --user flag."
  PIP="$PYTHON -m pip"
}

PIP="${PIP:-python -m pip}"
success "Using pip: $PIP"

# ── Write optimised requirements.txt ────────────────────────────
step "Writing optimised requirements.txt (CPU-only PyTorch for cloud)"
cat > requirements.txt << 'EOF'
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.0.1+cpu
torchvision==0.15.2+cpu
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.9
opencv-python-headless>=4.9.0
Pillow>=10.2.0
numpy>=1.26.0
scikit-learn>=1.4.0
tqdm>=4.66.0
pydantic>=2.6.0
EOF
success "requirements.txt updated (CPU torch — fits inside free tier memory)"

# ── Install packages ─────────────────────────────────────────────
step "Installing packages"
info "This may take 5–10 minutes on first run..."
$PIP install --upgrade pip --quiet
$PIP install -r requirements.txt
success "All packages installed"

# ── Patch main.py for dynamic PORT ──────────────────────────────
step "Patching main.py to support dynamic PORT (required for Railway/Heroku)"
if grep -q 'os.environ.get("PORT"' main.py; then
  warn "main.py already patched — skipping"
else
  # Use Python to patch the file safely
  $PYTHON - << 'PYEOF'
import re

with open("main.py", "r") as f:
    content = f.read()

old = 'uvicorn.run("app:app", host="0.0.0.0", port=args.port, reload=args.dev)'
new = '''import os
    port = int(os.environ.get("PORT", args.port))
    uvicorn.run("app:app", host="0.0.0.0", port=port)'''

if old in content:
    content = content.replace(old, new)
    with open("main.py", "w") as f:
        f.write(content)
    print("  Patched successfully")
else:
    print("  Pattern not found — please check main.py manually")
PYEOF
  success "main.py patched for dynamic PORT"
fi

# ── Create Procfile ──────────────────────────────────────────────
step "Creating Procfile"
cat > Procfile << 'EOF'
web: python main.py serve --port $PORT
EOF
success "Procfile created"

# ── Create railway.json ──────────────────────────────────────────
step "Creating railway.json"
cat > railway.json << 'EOF'
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python main.py serve --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 5
  }
}
EOF
success "railway.json created"

# ── Create runtime.txt ───────────────────────────────────────────
step "Creating runtime.txt"
echo "python-3.10.13" > runtime.txt
success "runtime.txt created"

# ── Create .gitignore ────────────────────────────────────────────
step "Creating .gitignore"
cat > .gitignore << 'EOF'
# Python
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/

# Model checkpoints (too large for git)
checkpoints/
*.pth
*.pt
*.ckpt

# Data
data/
*.csv
*.json.bak

# OS
.DS_Store
Thumbs.db

# VS Code
.vscode/settings.json
.vscode/launch.json

# Env
.env
.env.local

# Logs
*.log
logs/
results/
EOF
success ".gitignore created"

# ── Git setup ────────────────────────────────────────────────────
step "Setting up Git repository"

# Configure git user if not set
GIT_EMAIL=$(git config --global user.email 2>/dev/null || echo "")
GIT_NAME=$(git config --global user.name 2>/dev/null || echo "")

if [ -z "$GIT_EMAIL" ]; then
  echo -e "${YELLOW}Enter your GitHub email:${RESET} "
  read GIT_EMAIL
  git config --global user.email "$GIT_EMAIL"
fi

if [ -z "$GIT_NAME" ]; then
  echo -e "${YELLOW}Enter your name:${RESET} "
  read GIT_NAME
  git config --global user.name "$GIT_NAME"
fi

success "Git user: $GIT_NAME <$GIT_EMAIL>"

# Init if needed
if [ ! -d ".git" ]; then
  git init
  success "Git repository initialized"
else
  warn "Git already initialized — skipping init"
fi

# ── GitHub remote ────────────────────────────────────────────────
step "Connecting to GitHub"
EXISTING_REMOTE=$(git remote get-url origin 2>/dev/null || echo "")

if [ -z "$EXISTING_REMOTE" ]; then
  echo -e "${YELLOW}Enter your GitHub username:${RESET} "
  read GH_USERNAME
  echo -e "${YELLOW}Enter your GitHub repository name (e.g. deepfake-detection):${RESET} "
  read GH_REPO
  REMOTE_URL="https://github.com/$GH_USERNAME/$GH_REPO.git"
  git remote add origin "$REMOTE_URL"
  success "Remote set to $REMOTE_URL"
else
  success "Remote already set to $EXISTING_REMOTE"
fi

# ── Commit and push ──────────────────────────────────────────────
step "Committing and pushing to GitHub"
git add .
git commit -m "feat: add Railway deployment config and optimised requirements" 2>/dev/null || {
  warn "Nothing new to commit — already up to date"
}
git branch -M main
info "Pushing to GitHub..."
git push -u origin main
success "Code pushed to GitHub"

# ── Done ─────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}═══════════════════════════════════════════════════${RESET}"
echo -e "${GREEN}${BOLD}   ALL DONE! Your project is ready to deploy.${RESET}"
echo -e "${GREEN}${BOLD}═══════════════════════════════════════════════════${RESET}"
echo ""
echo -e "${BOLD}Next steps to go live on Railway:${RESET}"
echo ""
echo -e "  1. Go to ${CYAN}https://railway.app${RESET}"
echo -e "  2. Sign in with GitHub"
echo -e "  3. Click ${BOLD}New Project → Deploy from GitHub Repo${RESET}"
echo -e "  4. Select your ${BOLD}$GH_REPO${RESET} repository"
echo -e "  5. Railway auto-detects everything — click ${BOLD}Deploy${RESET}"
echo -e "  6. Your app will be live at a ${CYAN}*.railway.app${RESET} URL in ~5 minutes"
echo ""
echo -e "${BOLD}Useful commands:${RESET}"
echo -e "  ${CYAN}git add . && git commit -m 'update' && git push${RESET}  ← redeploy anytime"
echo -e "  ${CYAN}python main.py serve --port 8000${RESET}                 ← run locally"
echo ""
