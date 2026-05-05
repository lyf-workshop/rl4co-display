#!/usr/bin/env bash
# RL4CO Display - Auto Update Script (Linux/Mac deployment)
# Usage: bash scripts/update.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

step() { echo -e "${BLUE}[$1/6]${NC} $2"; }
ok()   { echo -e "  ${GREEN}[OK]${NC} $*"; }
warn() { echo -e "  ${YELLOW}[!!]${NC} $*"; }
err()  { echo -e "  ${RED}[ERR]${NC} $*"; }

echo ""
echo "================================================"
echo "   RL4CO Display - Auto Update"
echo "================================================"
echo ""

# ── 0. 前置检查 ───────────────────────────────────
if ! command -v git &>/dev/null; then
    err "git not found. Install git first."
    exit 1
fi

if [ ! -f "config/config.py" ]; then
    warn "config/config.py not found!"
    warn "Copy config/config.example.py and fill in your DB credentials."
    warn "The update will continue but the app may not start without it."
    echo ""
fi

# ── 1. 获取远程信息 ───────────────────────────────
step 1 "Fetching latest info from GitHub..."
if ! git fetch origin main 2>&1; then
    err "Cannot reach GitHub. Check network connection."
    exit 1
fi

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" = "$REMOTE" ]; then
    echo ""
    ok "Already up to date (${LOCAL:0:8}). No update needed."
    echo ""
    exit 0
fi

# ── 2. 显示即将拉取的提交 ─────────────────────────
echo ""
step 2 "New commits to apply:"
echo "  ------------------------------------------------"
git log --oneline HEAD..origin/main | sed 's/^/  /'
echo "  ------------------------------------------------"
echo ""

# ── 3. 暂存本地修改（不含 gitignored 文件） ────────
step 3 "Checking for local changes..."
STASHED=0
if ! git diff --quiet HEAD 2>/dev/null; then
    warn "Local changes detected, stashing..."
    git stash push -m "auto-stash before update $(date '+%Y-%m-%d %H:%M:%S')" 2>&1 | sed 's/^/  /'
    STASHED=1
else
    ok "No local changes"
fi

# 记录 requirements.txt 哈希
REQ_BEFORE=$(git hash-object requirements.txt 2>/dev/null || echo "")

# ── 4. 拉取最新代码 ───────────────────────────────
echo ""
step 4 "Pulling latest version..."
if ! git pull origin main; then
    err "git pull failed."
    [ $STASHED -eq 1 ] && git stash pop && warn "Local changes restored."
    exit 1
fi

if [ $STASHED -eq 1 ]; then
    echo "  Restoring local changes..."
    if ! git stash pop 2>&1; then
        warn "Conflict restoring local changes. Run 'git stash list' to inspect."
    else
        ok "Local changes restored"
    fi
fi

# ── 5. 依赖更新 ───────────────────────────────────
echo ""
step 5 "Checking dependencies..."
REQ_AFTER=$(git hash-object requirements.txt 2>/dev/null || echo "")
if [ -n "$REQ_BEFORE" ] && [ "$REQ_BEFORE" != "$REQ_AFTER" ]; then
    warn "requirements.txt changed, updating dependencies..."
    PYTHON="${PYTHON_CMD:-python3}"
    if $PYTHON -m pip install -r requirements.txt --quiet; then
        ok "Dependencies updated"
    else
        warn "pip install failed. Run manually: pip install -r requirements.txt"
    fi
else
    ok "Dependencies unchanged, skipping"
fi

# ── 6. 应用重启提示 ───────────────────────────────
echo ""
step 6 "Restart application"

# 尝试检测常见的进程管理方式
if systemctl is-active --quiet rl4co-display 2>/dev/null; then
    echo "  Detected systemd service 'rl4co-display', restarting..."
    systemctl restart rl4co-display
    ok "Service restarted via systemd"
elif systemctl is-active --quiet rl4co 2>/dev/null; then
    echo "  Detected systemd service 'rl4co', restarting..."
    systemctl restart rl4co
    ok "Service restarted via systemd"
else
    warn "Cannot auto-detect service manager."
    warn "Please restart the application manually:"
    warn "  pkill -f 'python app.py' && python app.py &"
    warn "  or: systemctl restart <your-service-name>"
fi

# ── 完成 ──────────────────────────────────────────
NEW_HEAD=$(git rev-parse HEAD)
LAST_MSG=$(git log -1 --format="%s (%ar)" HEAD)
echo ""
echo "================================================"
echo -e "  ${GREEN}Update complete!${NC}"
echo "  Version : ${NEW_HEAD:0:8}"
echo "  Latest  : $LAST_MSG"
echo "================================================"
echo ""
