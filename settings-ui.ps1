# Hidden launcher for the optional PySide/Qt settings UI.
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$venvPy = Join-Path $env:USERPROFILE 'voice-pi-venv\Scripts\python.exe'
$app = Join-Path $here 'voice_pi.py'
$uiReq = Join-Path $here 'requirements-ui.txt'
$logDir = Join-Path $env:APPDATA 'WhisperDictate'
$log = Join-Path $logDir 'settings-ui.log'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

try {
  if (-not (Test-Path $venvPy)) {
    powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File (Join-Path $here 'setup.ps1') --settings-ui
    exit $LASTEXITCODE
  }
  & $venvPy -c "import PySide6" 2>$null
  if ($LASTEXITCODE -ne 0 -and (Test-Path $uiReq)) {
    & $venvPy -m pip install -r $uiReq *>> $log
  }
  & $venvPy $app --settings-ui *>> $log
} catch {
  $_ | Out-File -FilePath $log -Append -Encoding utf8
}
