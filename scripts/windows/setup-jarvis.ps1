<#
.SYNOPSIS
    Install or update OpenJarvis from a GitHub repository and set it up as a
    desktop assistant (desktop icon, tray icon, "Hey Jarvis" at login).

.DESCRIPTION
    Paste this one line in PowerShell (no admin needed):

      irm https://raw.githubusercontent.com/JPnachh/OpenJarvis/main/scripts/windows/setup-jarvis.ps1 | iex

    Environment overrides:
      OPENJARVIS_REPO_URL  repository to install from (default: JPnachh/OpenJarvis)
      OPENJARVIS_BRANCH    branch to use (default: main)
      OPENJARVIS_HOME      install root (default: %LOCALAPPDATA%\OpenJarvis)

    The checkout lives in <root>\src, the same place the official installer
    uses, so an existing installation is switched over and updated in place.
#>

$ErrorActionPreference = 'Stop'

function Info($m) { Write-Host "  $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "  [ok] $m" -ForegroundColor Green }
function Fail($m) { Write-Host "  [x] $m" -ForegroundColor Red; throw $m }

$repo   = if ($env:OPENJARVIS_REPO_URL) { $env:OPENJARVIS_REPO_URL } else { 'https://github.com/JPnachh/OpenJarvis.git' }
$branch = if ($env:OPENJARVIS_BRANCH)   { $env:OPENJARVIS_BRANCH }   else { 'main' }
$root   = if ($env:OPENJARVIS_HOME)     { $env:OPENJARVIS_HOME }     else { Join-Path $env:LOCALAPPDATA 'OpenJarvis' }
$src    = Join-Path $root 'src'

Write-Host "`nOpenJarvis desktop setup" -ForegroundColor White
Info "Repository: $repo ($branch)"
Info "Folder:     $src"

# --- tools ------------------------------------------------------------------
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail "git is missing. Install it with:  winget install Git.Git   then open a new PowerShell and run this again."
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Info "Installing uv (Python package manager)..."
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Fail "uv did not install; see https://docs.astral.sh/uv/" }
}
Ok "git and uv found"
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "  [!] Node.js is missing (needed for the graphical interface)." -ForegroundColor Yellow
    Write-Host "      Install it with:  winget install OpenJS.NodeJS.LTS   then run this line again." -ForegroundColor Yellow
}

# --- code -------------------------------------------------------------------
if (-not (Test-Path $root)) { New-Item -ItemType Directory -Path $root | Out-Null }
if (Test-Path (Join-Path $src '.git')) {
    Info "Updating the existing checkout..."
    git -C $src remote set-url origin $repo
    git -C $src fetch origin $branch
    if ($LASTEXITCODE -ne 0) { Fail "git fetch failed" }
    git -C $src checkout -B $branch "origin/$branch"
    if ($LASTEXITCODE -ne 0) { Fail "git checkout failed (local edits in $src? commit or discard them)" }
} else {
    Info "Downloading OpenJarvis..."
    git clone --branch $branch $repo $src
    if ($LASTEXITCODE -ne 0) { Fail "git clone failed" }
}
Ok "Code is up to date"

# --- Python packages and desktop setup --------------------------------------
Push-Location $src
try {
    Info "Installing Python packages (first time takes a few minutes)..."
    uv sync --extra desktop
    if ($LASTEXITCODE -ne 0) { Fail "uv sync failed; see the output above" }
    Ok "Python packages installed"
    uv run jarvis setup-desktop
} finally {
    Pop-Location
}

Write-Host "`nDone. Double-click 'OpenJarvis' on your desktop." -ForegroundColor Green
Write-Host "Folder: $src  (run 'uv run jarvis ...' commands from there)`n"
