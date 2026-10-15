# Runs ONLY in this new V9 folder. No commands target or modify mb_v8.
[CmdletBinding()]
param(
    [string]$Owner = 'prestigegitserp',
    [string]$RepoName = 'mb_v9',
    [switch]$Public
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (!(Test-Path './vendor_v8/mbmf_q1_final_v7.py')) {
    throw 'First run bootstrap_v8.py to copy the pinned V8 scientific files to this NEW folder.'
}
if (!(Get-Command gh -ErrorAction SilentlyContinue)) { throw 'GitHub CLI (gh) is missing; install it with: winget install GitHub.cli' }
& gh auth status
if ($LASTEXITCODE -ne 0) { throw 'Authenticate first: gh auth login' }
if (!(Test-Path '.git')) { & git init -b main; if ($LASTEXITCODE -ne 0) { throw 'git init failed' } }
$remote = & git remote get-url origin 2>$null
if ($LASTEXITCODE -eq 0 -and $remote -and ($remote -notmatch "(^|[/:])$([regex]::Escape($Owner))/$([regex]::Escape($RepoName))(.git)?$")) {
    throw "Safety stop: origin points to a different repository: $remote"
}
& git add .
if ($LASTEXITCODE -ne 0) { throw 'git add failed' }
& git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    & git commit -m 'MBMF V9: audited validation-first experiments and progress reporting'
    if ($LASTEXITCODE -ne 0) { throw 'git commit failed; check git user.name/user.email' }
}
if (!$remote) {
    $visibility = if ($Public) { '--public' } else { '--private' }
    & gh repo create "$Owner/$RepoName" $visibility --source . --remote origin --push
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create new GitHub repository; no changes made to mb_v8' }
} else {
    & git push -u origin HEAD:main
    if ($LASTEXITCODE -ne 0) { throw 'Failed to push new V9 repository' }
}
Write-Host "V9 published: https://github.com/$Owner/$RepoName"
