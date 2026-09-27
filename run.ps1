$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$localPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$preparedPython = Join-Path $PSScriptRoot '../../work/agent-venv/Scripts/python.exe'
if (Test-Path -LiteralPath $localPython) {
    $agentPython = $localPython
} elseif (Test-Path -LiteralPath $preparedPython) {
    $agentPython = (Resolve-Path -LiteralPath $preparedPython).Path
} else {
    $bundledPython = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
    if (Test-Path -LiteralPath $bundledPython) {
        & $bundledPython -m venv .venv
    } else {
        & py -3 -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.10 or later is required. Install Python, then open START.cmd again.' }
    $agentPython = $localPython
}
& $agentPython -c 'import gradio, httpx, pypdf, bs4' 2>$null
if ($LASTEXITCODE -ne 0) {
    & $agentPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check the connection and try again.' }
}
& $agentPython app.py
