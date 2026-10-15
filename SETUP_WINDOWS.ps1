[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$V8Source,
  [switch]$SkipInstall,
  [switch]$AllowUnpinned
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (!$SkipInstall) {
  python -m pip install -r requirements.txt
  if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }
}
$boot = @('bootstrap_v8.py','--from-v8',$V8Source)
if ($AllowUnpinned) { $boot += '--allow-unpinned' }
python @boot
if ($LASTEXITCODE -ne 0) { throw 'V8 scientific source bootstrap failed' }
python -m pytest -q tests/test_v9_validation.py
if ($LASTEXITCODE -ne 0) { throw 'V9 safety tests failed' }
python run_v9.py self-test
if ($LASTEXITCODE -ne 0) { throw 'V8 scientific regression tests failed' }
Write-Host 'V9 setup completed. Try: python run_v9.py plan --datasets ml-100k'
