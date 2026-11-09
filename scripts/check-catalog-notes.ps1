#!/usr/bin/env pwsh
<#
.SYNOPSIS
  Reports a catalog change that has no CHANGELOG line.

.DESCRIPTION
  tests/LMSupply.Generator.Tests/catalog.snapshot records what every catalog alias resolves to, and
  CatalogSnapshotTests keeps it equal to the registries. This script compares the snapshot with the one at -Base and,
  for every alias whose line was added, removed or changed, requires a line added to CHANGELOG.md in the same range
  that names the alias (for example `qwen3-balanced: Qwen3 8B -> Qwen 3.5 9B (why)`). A repoint, a changed default
  file or quantization, a changed known-issue tag, a retirement and a change of the automatic-selection pool all count.

  Covers the generator catalogs (GGUF and ONNX). docs/CATALOG.md describes the rule.

.PARAMETER Base
  The revision to compare with. Default: the upstream of the current branch, else origin/main.

.EXAMPLE
  pwsh scripts/check-catalog-notes.ps1            # before pushing
  pwsh scripts/check-catalog-notes.ps1 -Base v0.140.0

  Exit code: 0 = no catalog change, or every change has a line · 1 = a change without a line · 2 = could not compare.
#>
param(
    [string]$Base = '',
    [string]$Snapshot = 'tests/LMSupply.Generator.Tests/catalog.snapshot',
    [string]$Changelog = 'CHANGELOG.md'
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

function Get-Rows([string[]]$lines) {
    $rows = @{}
    foreach ($l in $lines) {
        if ([string]::IsNullOrWhiteSpace($l) -or $l.StartsWith('#')) { continue }
        $rows[($l -split ' \| ', 2)[0].Trim()] = $l.TrimEnd()
    }
    $rows
}

if (-not $Base) {
    $Base = (git rev-parse --abbrev-ref '@{u}' 2>$null)
    if ($LASTEXITCODE -ne 0 -or -not $Base) { $Base = 'origin/main' }
}
git rev-parse --verify --quiet "$Base^{commit}" *> $null
if ($LASTEXITCODE -ne 0) { Write-Host "catalog notes: cannot resolve '$Base'"; exit 2 }

if (-not (Test-Path $Snapshot)) { Write-Host "catalog notes: $Snapshot not found"; exit 2 }
git cat-file -e "${Base}:$Snapshot" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "catalog notes: $Base has no snapshot - this range introduces it, nothing to compare"; exit 0 }
$old = Get-Rows (git show "${Base}:$Snapshot")
$new = Get-Rows (Get-Content $Snapshot)

$changed = @($old.Keys + $new.Keys | Sort-Object -Unique | Where-Object { $old[$_] -ne $new[$_] })
if ($changed.Count -eq 0) { Write-Host "catalog notes: no catalog change since $Base"; exit 0 }

$added = @(git diff -U0 $Base -- $Changelog | Where-Object { $_.StartsWith('+') -and -not $_.StartsWith('+++') })
$missing = @()
foreach ($key in $changed) {
    $short = $key -replace '^(gguf|onnx):', ''
    $pattern = '(?<![\w:./-])(' + [regex]::Escape($key) + '|' + [regex]::Escape($short) + ')(?![\w-])'
    if (-not ($added | Where-Object { $_ -match $pattern })) { $missing += $key }
}

if ($missing.Count -gt 0) {
    Write-Host "catalog notes: $($missing.Count) changed alias(es) without a CHANGELOG line added since ${Base}:"
    foreach ($m in $missing) { Write-Host "  $m" }
    Write-Host "Add one line per alias to the unreleased section, e.g. '<alias>: <old> -> <new> (why)'. See docs/CATALOG.md."
    exit 1
}
Write-Host "catalog notes: $($changed.Count) changed alias(es), each named in a CHANGELOG line"
exit 0
