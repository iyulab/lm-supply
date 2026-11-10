#!/usr/bin/env pwsh
<#
.SYNOPSIS
  Reports a catalog change that has no CHANGELOG line.

.DESCRIPTION
  The catalog.snapshot of every domain's test project (generator, embedder, reranker, transcriber, synthesizer, OCR,
  translator, captioner, detector, segmenter, image generator) records what every catalog alias resolves to, and each
  project's CatalogSnapshotTests keeps it equal to the registries. This script compares each snapshot with
  the one at -Base and, for every alias whose line was added, removed or changed, requires a line added to CHANGELOG.md
  in the same range that names the alias (for example `qwen3-balanced: Qwen3 8B -> Qwen 3.5 9B (why)`). A repoint, a
  changed default file or quantization, a changed known-issue tag, a retirement and a change of the pool or tier mapping
  `auto` draws from all count. An alias outside the generator's is named with its domain (`embedder:default`, or `default`
  on a line that says embedder), because those short names are common words.

  Covers every domain's catalog. docs/CATALOG.md describes the rule.

.PARAMETER Base
  The revision to compare with. Default: the upstream of the current branch, else origin/main.

.EXAMPLE
  pwsh scripts/check-catalog-notes.ps1            # before pushing
  pwsh scripts/check-catalog-notes.ps1 -Base v0.140.0

  Exit code: 0 = no catalog change, or every change has a line · 1 = a change without a line · 2 = could not compare.
#>
param(
    [string]$Base = '',
    [string[]]$Snapshot = @(
        'tests/LMSupply.Generator.Tests/catalog.snapshot',
        'tests/LMSupply.Embedder.Tests/catalog.snapshot',
        'tests/LMSupply.Reranker.Tests/catalog.snapshot',
        'tests/LMSupply.Transcriber.Tests/catalog.snapshot',
        'tests/LMSupply.Synthesizer.Tests/catalog.snapshot',
        'tests/LMSupply.Ocr.Tests/catalog.snapshot',
        'tests/LMSupply.Translator.Tests/catalog.snapshot',
        'tests/LMSupply.Captioner.Tests/catalog.snapshot',
        'tests/LMSupply.Detector.Tests/catalog.snapshot',
        'tests/LMSupply.Segmenter.Tests/catalog.snapshot',
        'tests/LMSupply.ImageGenerator.Tests/catalog.snapshot'),
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

$changed = @()
foreach ($file in $Snapshot) {
    if (-not (Test-Path $file)) { Write-Host "catalog notes: $file not found"; exit 2 }
    git cat-file -e "${Base}:$file" 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Host "catalog notes: $Base has no $file - this range introduces it, nothing to compare"; continue }
    $old = Get-Rows (git show "${Base}:$file")
    $new = Get-Rows (Get-Content $file)
    $changed += @($old.Keys + $new.Keys | Sort-Object -Unique | Where-Object { $old[$_] -ne $new[$_] })
}
if ($changed.Count -eq 0) { Write-Host "catalog notes: no catalog change since $Base"; exit 0 }

$added = @(git diff -U0 $Base -- $Changelog | Where-Object { $_.StartsWith('+') -and -not $_.StartsWith('+++') })
$missing = @()
foreach ($key in $changed) {
    $domain = ($key -split ':', 2)[0]
    $short = $key -replace '^[a-z-]+:', ''
    $full = '(?<![\w:./-])' + [regex]::Escape($key) + '(?![\w-])'
    $bare = '(?<![\w:./-])' + [regex]::Escape($short) + '(?![\w-])'
    # gguf and onnx aliases are named by their short form too; the other domains' only beside their domain (`ocr` for both OCR catalogs)
    $word = if ($domain -like 'ocr-*') { 'ocr' } else { $domain -replace '-', '[- ]' }
    $named = {
        param($line)
        if ($line -match $full) { return $true }
        if ($domain -notin 'gguf', 'onnx') { return ($line -match $bare) -and ($line -match "(?i)\b$word\b") }
        return $line -match $bare
    }
    if (-not ($added | Where-Object { & $named $_ })) { $missing += $key }
}

if ($missing.Count -gt 0) {
    Write-Host "catalog notes: $($missing.Count) changed alias(es) without a CHANGELOG line added since ${Base}:"
    foreach ($m in $missing) { Write-Host "  $m" }
    Write-Host "Add one line per alias to the unreleased section, e.g. '<alias>: <old> -> <new> (why)'. See docs/CATALOG.md."
    exit 1
}
Write-Host "catalog notes: $($changed.Count) changed alias(es), each named in a CHANGELOG line"
exit 0
