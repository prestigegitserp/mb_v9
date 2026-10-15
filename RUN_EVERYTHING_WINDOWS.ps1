[CmdletBinding()]
param(
    [string]$Datasets='synthetic,jester,bookcrossing,ml-100k,ml-1m,ml-10m',
    [string]$K='5,10,20,50,64',
    [string]$DataRoot='./datasets',
    [string]$Output='./results_v9',
    [ValidateSet('one-factor','joint','hybrid')][string]$ScanMode='hybrid',
    [switch]$Resume,
    [switch]$DryRun,
    [switch]$EpochTrainRMSE,
    [switch]$IncludeTemporal,
    [switch]$IncludeConvergence
)
$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
$a=@('run_v9.py','all','--datasets',$Datasets,'--k',$K,'--data-root',$DataRoot,'--output',$Output,'--scan-mode',$ScanMode)
if ($Resume) { $a+='--resume' }
if ($DryRun) { $a+='--dry-run' }
if ($EpochTrainRMSE) { $a+='--epoch-train-rmse' }
if ($IncludeTemporal) { $a+='--include-temporal' }
if ($IncludeConvergence) { $a+='--include-convergence' }
python @a
if ($LASTEXITCODE -ne 0) { throw 'V9 run stopped. Check per-job V9_JOB_STATUS.json; restart with -Resume.' }
