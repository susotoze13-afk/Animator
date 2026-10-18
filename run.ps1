<#
.SYNOPSIS
  One-click launcher for the Animator whiteboard-explainer engine.

.DESCRIPTION
  Wraps `python -m animator` so you can go from an idea to an MP4 with one command:
  creates/repairs the venv, checks for an API key, picks an output path and renders.

  Prompt mode (needs a key in .env, or the keyless opencode backend):
      .\run.ps1 -Prompt "How do planes fly?"
      .\run.ps1 -Prompt "How bees make honey" -Scenes 4 -Quality high -SaveStoryboard

      Set ANIMATOR_BACKEND=opencode in .env to run on your OpenCode stack
      (Pollinations key from the OpenCode image tool + Windows speech): no
      Google/OpenRouter key needed.

  Image mode (offline, no key) — any positional args are passed straight to the engine:
      .\run.ps1 scene.png --duration 8
      .\run.ps1 a.png b.png --audio a.wav b.wav

  No arguments → asks for an idea interactively.

.EXAMPLE
  .\run.ps1
#>
[CmdletBinding()]
param(
  [string]$Prompt,
  [int]$Scenes = 0,
  [string]$Voice = '',
  [ValidateSet('low', 'medium', 'high')]$Quality = 'medium',
  [string]$Output = '',
  [switch]$SaveStoryboard,
  [switch]$Quiet,
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Rest
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$py = Join-Path $root '.venv\Scripts\python.exe'

function Ensure-Env {
  if (Test-Path $py) { return }
  Write-Host 'First run: creating venv and installing dependencies...' -ForegroundColor Cyan
  python -m venv (Join-Path $root '.venv')
  & $py -m pip install --upgrade pip
  & $py -m pip install -r (Join-Path $root 'requirements.txt')
  if (-not (Test-Path $py)) { throw 'Could not create the venv (is python on PATH?)' }
}

function Test-Key {
  if ($env:GEMINI_API_KEY -or $env:GOOGLE_API_KEY -or $env:OPENROUTER_API_KEY -or $env:POLLINATIONS_API_KEY) { return $true }
  $envFile = Join-Path $root '.env'
  if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile) {
      if ($line -match '^\s*(GEMINI_API_KEY|GOOGLE_API_KEY|OPENROUTER_API_KEY|POLLINATIONS_API_KEY)\s*=\s*\S+') { return $true }
      if ($line -match '^\s*ANIMATOR_BACKEND\s*=\s*opencode\s*$') { return $true }
    }
  }
  # The opencode backend needs no Google/OpenRouter key: it reuses the
  # Pollinations key OpenCode's image tool already has and narrates with the OS.
  $token = Join-Path $env:USERPROFILE '.config\opencode\pollinations-shim\token.txt'
  if (Test-Path $token) { return $true }
  return $false
}

function New-Slug([string]$text) {
  $s = $text.ToLower()
  $s = [regex]::Replace($s, '[^\p{L}\p{Nd}]+', '-').Trim('-')
  if (-not $s) { $s = 'video' }
  if ($s.Length -gt 40) { $s = $s.Substring(0, 40).Trim('-') }
  return $s
}

# PowerShell binds the first free positional argument to $Prompt. When that
# argument is really an image path or a raw flag (image mode), move it back to
# the passthrough list so the engine sees it verbatim.
if ($Prompt -and ($Prompt.StartsWith('-') -or (Test-Path $Prompt))) {
  $Rest = @($Prompt) + $Rest
  $Prompt = $null
}

Ensure-Env
New-Item -ItemType Directory -Force -Path (Join-Path $root 'outputs') | Out-Null

if (-not $Prompt -and -not $Rest) {
  Write-Host 'Animator - whiteboard explainer videos' -ForegroundColor Cyan
  $Prompt = Read-Host 'Idea / topic (e.g. How do planes fly?)'
  if (-not $Prompt) { Write-Host 'Cancelled.'; exit 0 }
}

$cliArgs = @()

if ($Prompt) {
  if (-not (Test-Key)) {
    Write-Host 'No API key found.' -ForegroundColor Yellow
    Write-Host 'Add one of these to .env, or rely on the keyless opencode backend:' -ForegroundColor Yellow
    Write-Host '  ANIMATOR_BACKEND=opencode   (reuses your OpenCode Pollinations key + Windows speech)' -ForegroundColor Yellow
    Write-Host ("and add them to:  {0}\.env" -f $root) -ForegroundColor Yellow
    Write-Host '  OPENROUTER_API_KEY=your-key   (runs the whole pipeline on OpenRouter)' -ForegroundColor Yellow
    Write-Host '  GEMINI_API_KEY=your-key       (Google AI Studio key)' -ForegroundColor Yellow
    Write-Host 'Or use image mode offline:  .\run.ps1 <image.png> [--audio <wav>]' -ForegroundColor Yellow
    exit 2
  }
  $cliArgs += '--prompt', $Prompt
  if ($Scenes -gt 0) { $cliArgs += '--scenes', $Scenes }
  if ($Voice) { $cliArgs += '--voice', $Voice }
  if ($SaveStoryboard) { $cliArgs += '--save-storyboard' }
}

# positional passthrough (image mode) and any extra raw flags
if ($Rest) { $cliArgs += $Rest }

if (-not $Output) {
  $base = if ($Prompt) { New-Slug $Prompt } else { 'video' }
  $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
  $Output = Join-Path $root ("outputs\{0}-{1}.mp4" -f $base, $stamp)
}
$cliArgs += '-o', $Output
$cliArgs += '--quality', $Quality
if (-not $Quiet) { $cliArgs += '-v' }

Write-Host ('Running: python -m animator {0}' -f ($cliArgs -join ' ')) -ForegroundColor Cyan
& $py -m animator @cliArgs
$code = $LASTEXITCODE
if ($code -ne 0) {
  Write-Host ("Animator failed (exit {0})." -f $code) -ForegroundColor Red
  exit $code
}

if (Test-Path $Output) {
  $duration = ''
  try { $duration = & ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 $Output 2>$null } catch {}
  $line = ('Done: {0}' -f (Resolve-Path $Output).Path)
  if ($duration) { $line += (' ({0:N1}s)' -f [double]$duration) }
  Write-Host $line -ForegroundColor Green
}
