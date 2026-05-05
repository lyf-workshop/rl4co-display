# RL4CO Display - Auto Update Script
# Usage: powershell -ExecutionPolicy Bypass -File scripts\update.ps1

# Switch to project root (script is in scripts/ subdirectory)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

function Write-Step($n, $msg) { Write-Host "[$n/5] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)        { Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Warn($msg)      { Write-Host "  [!!] $msg" -ForegroundColor Yellow }
function Write-Err($msg)       { Write-Host " [ERR] $msg" -ForegroundColor Red }

Write-Host ""
Write-Host "================================================" -ForegroundColor Blue
Write-Host "   RL4CO Display - Auto Update" -ForegroundColor Blue
Write-Host "================================================" -ForegroundColor Blue
Write-Host ""

# Check git
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Err "git not found. Please install Git for Windows."
    Read-Host "Press Enter to exit"; exit 1
}

# Check git repo
$isRepo = git rev-parse --git-dir 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Err "Not a git repository: $ProjectDir"
    Read-Host "Press Enter to exit"; exit 1
}

# Step 1: Fetch
Write-Step 1 "Fetching latest info from GitHub..."
git fetch origin main
if ($LASTEXITCODE -ne 0) {
    Write-Err "git fetch failed. Check network or GitHub access."
    Read-Host "Press Enter to exit"; exit 1
}

$local  = git rev-parse HEAD
$remote = git rev-parse origin/main

if ($local -eq $remote) {
    Write-Host ""
    Write-Ok "Already up to date. No update needed."
    Write-Host "  Current: $($local.Substring(0,8))" -ForegroundColor Gray
    Write-Host ""
    Read-Host "Press Enter to exit"; exit 0
}

# Step 2: Show incoming commits
Write-Host ""
Write-Step 2 "New commits to be applied:"
Write-Host "  ------------------------------------------------" -ForegroundColor Gray
git log --oneline HEAD..origin/main | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
Write-Host "  ------------------------------------------------" -ForegroundColor Gray
Write-Host ""

# Step 3: Stash local changes
Write-Step 3 "Checking for local changes..."
$status = git status --porcelain
$stashed = $false
if ($status) {
    Write-Warn "Local changes detected, stashing..."
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    git stash push -m "auto-stash before update $timestamp"
    $stashed = $true
    Write-Ok "Local changes stashed"
} else {
    Write-Ok "No local changes"
}

# Record requirements.txt hash before update
$reqBefore = ""
if (Test-Path "requirements.txt") {
    $reqBefore = (Get-FileHash "requirements.txt" -Algorithm SHA256).Hash
}

# Step 4: Pull
Write-Host ""
Write-Step 4 "Pulling latest version..."
git pull origin main
if ($LASTEXITCODE -ne 0) {
    Write-Err "git pull failed."
    if ($stashed) {
        git stash pop
        Write-Warn "Local changes restored from stash."
    }
    Read-Host "Press Enter to exit"; exit 1
}

# Restore stash
if ($stashed) {
    Write-Host ""
    Write-Host "  Restoring local changes..." -ForegroundColor Cyan
    git stash pop
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Conflict restoring local changes."
        Write-Warn "Run 'git stash list' to inspect stashed content."
    } else {
        Write-Ok "Local changes restored"
    }
}

# Step 5: Check requirements
Write-Host ""
Write-Step 5 "Checking dependencies..."
if (Test-Path "requirements.txt") {
    $reqAfter = (Get-FileHash "requirements.txt" -Algorithm SHA256).Hash
    if ($reqBefore -ne $reqAfter) {
        Write-Warn "requirements.txt changed, updating dependencies..."
        pip install -r requirements.txt --quiet
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "Dependencies updated"
        } else {
            Write-Warn "pip install failed. Run manually: pip install -r requirements.txt"
        }
    } else {
        Write-Ok "Dependencies unchanged, skipping"
    }
}

# Summary
$newHead = git rev-parse HEAD
$lastMsg = git log -1 --format="%s (%ar)" HEAD
Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "   Update complete!" -ForegroundColor Green
Write-Host "   Version : $($newHead.Substring(0,8))" -ForegroundColor Gray
Write-Host "   Latest  : $lastMsg" -ForegroundColor Gray
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Please restart the application to apply updates." -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter to exit"
