#!/usr/bin/env bash
# deploy.sh
# ─────────────────────────────────────────────────────────────────────────────
# Run once on a fresh Ubuntu 24.04 DigitalOcean Droplet.
# Installs Docker, clones your repo, and starts the app.
#
# Usage (as root on the Droplet):
#   chmod +x deploy.sh && ./deploy.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Must run as root
if [[ $EUID -ne 0 ]]; then
  echo "Run this script as root: sudo ./deploy.sh"
  exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  F1 Strategy Lab — Server Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. Install Docker ──────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
  echo "[1/7] Installing Docker..."
  apt-get update -q
  apt-get install -y -q ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -q
  apt-get install -y -q docker-ce docker-ce-cli containerd.io docker-compose-plugin
  systemctl enable docker
  systemctl start docker
  echo "  ✓ Docker installed"
else
  echo "[1/7] Docker already installed — skipping"
fi

# ── 2. Basic firewall ──────────────────────────────────────────────────────────
echo "[2/7] Configuring UFW firewall..."
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 443/udp   # HTTP/3
ufw --force enable
echo "  ✓ Firewall configured (ssh, 80, 443)"

# ── 3. Create app directory ────────────────────────────────────────────────────
echo "[3/7] Creating /opt/f1app..."
mkdir -p /opt/f1app
cd /opt/f1app

# ── 4. Clone / pull repo ───────────────────────────────────────────────────────
echo "[4/7] Pulling latest code..."
REPO_URL="${REPO_URL:-https://github.com/YOUR_USERNAME/f1-strategy-app.git}"

if [ -d ".git" ]; then
  git pull origin main
else
  git clone "$REPO_URL" .
fi
echo "  ✓ Code ready"

# ── 5. Model artifacts ─────────────────────────────────────────────────────────
echo "[5/7] Checking model artifacts..."
mkdir -p data/trained_models data/feature_data data/championship

MISSING=0
for f in \
  "data/trained_models/best_lap_time_predictor.pth" \
  "data/trained_models/f1_preprocessors.joblib" \
  "data/feature_data/features_dataset_with_targets.csv"; do
  if [ ! -f "$f" ]; then
    echo "  ✗ Missing: $f"
    MISSING=1
  fi
done

if [ $MISSING -eq 1 ]; then
  echo ""
  echo "  ⚠️  Model files missing. From your LOCAL machine, run:"
  echo ""
  echo "    scp -r ./data root@$(curl -s ifconfig.me 2>/dev/null || echo YOUR_IP):/opt/f1app/"
  echo ""
  echo "  Then re-run this script."
  echo ""
  read -r -p "  Continue anyway (app won't start without them)? [y/N] " response
  [[ "$response" =~ ^[Yy]$ ]] || exit 1
else
  echo "  ✓ Model artifacts present"
fi

# ── 6. Write .env ──────────────────────────────────────────────────────────────
echo "[6/7] Configuring environment..."
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo ""
  echo "  ┌─────────────────────────────────────────────────┐"
  echo "  │  Set your domain in .env before continuing:     │"
  echo "  │    nano /opt/f1app/.env                         │"
  echo "  │  Example: DOMAIN=f1strategy.yourdomain.com      │"
  echo "  └─────────────────────────────────────────────────┘"
  echo ""
  read -r -p "  Press Enter once .env is configured..."
fi
echo "  ✓ Domain: $(grep DOMAIN .env | cut -d= -f2)"

# ── 7. Build and start ─────────────────────────────────────────────────────────
echo "[7/7] Building images and starting services (this takes ~3 min first time)..."
docker compose -f docker-compose.yml -f docker-compose.prod.yml build --no-cache
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅  Deployment complete!"
echo ""
DOMAIN_VAL=$(grep DOMAIN .env | cut -d= -f2)
echo "  🌐  App:       https://${DOMAIN_VAL}"
echo "  📋  API docs:  https://${DOMAIN_VAL}/api/docs"
echo ""
echo "  Useful commands:"
echo "    docker compose logs -f                    # tail all logs"
echo "    docker compose logs -f backend            # backend only"
echo "    docker compose restart backend            # restart after code change"
echo "    docker compose down                       # stop everything"
echo "    docker compose -f docker-compose.yml \\"
echo "      -f docker-compose.prod.yml up -d        # start again"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"