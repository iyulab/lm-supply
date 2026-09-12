# LMSupply test runner (CI and local)
#
# Enumerates test projects by discovery and requires every one of them to pass.
#
# Why this is a script and not an inline workflow block: the same twenty lines of shell lived in
# two workflows, and two copies of one rule drift apart without either side noticing. They also
# could not be run locally, so the command that gated a release was not a command anyone could
# reproduce on a development machine.
#
# What the inline block could not do, and this can: it ran the whole solution in one invocation and
# read a single aggregate summary. An aggregate cannot see one assembly disappear - a deleted file,
# an over-applied trait, a mistyped filter - because the other seventeen keep the totals healthy.
# Running project by project is what makes "this project contributed nothing" a statement the
# runner can make at all.

param(
    [ValidateSet("quiet", "minimal", "normal", "detailed", "diagnostic")]
    [string]$Verbosity = "normal",
    [switch]$NoBuild,
    [string]$Configuration = "Release",
    # Substring match on the project name, for narrowing a local run. Never used by CI: a filtered
    # run is not the gate, and this script says so in its summary when the switch is present.
    [string]$Filter
)

$ErrorActionPreference = "Stop"

function Write-Status {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

Write-Status "`n===================================" "Cyan"
Write-Status "LMSupply Test Runner" "Cyan"
Write-Status "===================================" "Cyan"
Write-Output ""

# Discovery, not an allowlist: a list cannot notice a project that was added, and hides one that
# was deleted behind a warning nobody reads. Not every test project here is named "*.Tests" - the
# pattern is every project under tests/.
$repoRoot = Split-Path -Parent $PSScriptRoot
$testsRoot = Join-Path $repoRoot "tests"

if (-not (Test-Path $testsRoot)) {
    Write-Status "ERROR: no tests/ directory at $testsRoot." "Red"
    exit 1
}

$testProjects = @(
    Get-ChildItem -Path $testsRoot -Filter "*.csproj" -Recurse -File |
        ForEach-Object { $_.FullName } |
        Sort-Object
)

if ($testProjects.Count -eq 0) {
    Write-Status "ERROR: no test projects discovered under tests/." "Red"
    Write-Output "Discovery returning nothing means a broken path or a moved tree, never 'nothing to test'."
    exit 1
}

# Projects whose every test needs something a shared runner does not have, so filtering the excluded
# categories out legitimately leaves nothing to run. Naming them is what lets zero tests be a failure
# everywhere else: without this list the runner has to treat "no tests matched" as normal for any
# project, and a suite that vanishes then reports green.
#
# Entries are structural, not waivers, so they carry no expiry: a project either has tests that run
# under this filter or it does not. Adding one means asserting that every test in it needs hardware
# or a service - check before you do.
$hardwareOnlyProjects = @(
    "TranscriberGpuTest"   # needs a GPU execution provider; deliberately outside the solution build
)

# A name that matches nothing is rot: the project was renamed or deleted and the entry now excuses a
# project that does not exist while covering nothing.
$discoveredNames = $testProjects | ForEach-Object { [System.IO.Path]::GetFileNameWithoutExtension($_) }
$staleEntries = @($hardwareOnlyProjects | Where-Object { $discoveredNames -notcontains $_ })
if ($staleEntries.Count -gt 0) {
    Write-Status "ERROR: `$hardwareOnlyProjects names project(s) that no longer exist: $($staleEntries -join ', ')" "Red"
    Write-Output "Remove the entry, or fix the name. An entry matching nothing silently excuses nothing."
    exit 1
}

if ($Filter) {
    $testProjects = @($testProjects | Where-Object { [System.IO.Path]::GetFileNameWithoutExtension($_) -like "*$Filter*" })
    if ($testProjects.Count -eq 0) {
        Write-Status "ERROR: -Filter '$Filter' matched none of the discovered projects." "Red"
        exit 1
    }
}

Write-Output "Discovered $($testProjects.Count) test project(s):"
$testProjects | ForEach-Object { Write-Output "  $(Split-Path -Leaf $_)" }
Write-Output ""

# Categories excluded here are excluded for a reason that holds on any machine:
#   Integration - needs a real model, a downloaded runtime, or a live endpoint
#   Functional  - same, end to end
#   LocalOnly   - reads a developer machine's real model cache
# Everything else must pass. Excluding by category rather than by project also covers the case an
# allowlist cannot express: a hardware-dependent test living inside an otherwise self-contained
# project.
#
# --filter-not-trait (repeated, ANDs together) is MTP's equivalent of VSTest's compound
# "Category!=A&Category!=B" filter syntax. --hangdump/--hangdump-timeout is the equivalent of
# --blame-hang: without it a stalled test is absorbed by the job timeout and surfaces only as an
# unexplained cancellation.
# --no-ansi because the counts below are read out of this output. MTP colours the summary, and it
# colours only the non-zero numbers - "succeeded: 0" arrives bare while "succeeded: 1809" arrives
# as an escape sequence followed by the number. A parser anchored at the start of the line can
# therefore read zero and nothing else, which is the one value that makes the zero-tests guard
# below fire.
$testArgs = @(
    "test",
    "--verbosity", $Verbosity,
    "--configuration", $Configuration,
    "--filter-not-trait", "Category=Integration",
    "--filter-not-trait", "Category=Functional",
    "--filter-not-trait", "Category=LocalOnly",
    "--hangdump", "--hangdump-timeout", "5m",
    "--no-ansi"
)
if ($NoBuild) { $testArgs += "--no-build" }

$allResults = @()
$totalPassed = 0
$totalFailed = 0
$totalSkipped = 0
$totalTests = 0
$failedProjects = @()

foreach ($project in $testProjects) {
    $projectName = [System.IO.Path]::GetFileNameWithoutExtension($project)

    Write-Status "`nRunning tests for: $projectName" "Yellow"
    Write-Output "-----------------------------------"

    # MTP's dotnet-test driver wants the project via --project, not as a trailing positional
    # (VSTest accepted the latter; MTP native mode rejects it).
    $testOutput = & dotnet @testArgs --project $project 2>&1
    $exitCode = $LASTEXITCODE
    $testOutput | ForEach-Object { Write-Output $_ }

    # MTP's summary is multi-line ("Test run summary: ..." followed by indented total:/failed:/
    # succeeded:/skipped: lines), unlike VSTest's single combined line.
    $passed = 0; $failed = 0; $skipped = 0; $total = 0
    $sawSummary = $false; $readTotal = $false
    foreach ($line in $testOutput) {
        if ($line -match "Test run summary:")              { $sawSummary = $true }
        if ($line -match "^\s*total:\s+(\d+)\s*$")         { $total = [int]$matches[1]; $readTotal = $true }
        elseif ($line -match "^\s*failed:\s+(\d+)\s*$")    { $failed = [int]$matches[1] }
        elseif ($line -match "^\s*succeeded:\s+(\d+)\s*$") { $passed = [int]$matches[1] }
        elseif ($line -match "^\s*skipped:\s+(\d+)\s*$")   { $skipped = [int]$matches[1] }
    }

    # "I could not read the counts" and "the counts were zero" are different facts, and only the
    # second one means anything about the tests. Collapsing the first into the second is how a
    # parsing bug arrives disguised as a vanished suite.
    if ($sawSummary -and -not $readTotal) {
        Write-Status "Result: COULD NOT READ THE TEST COUNTS" "Red"
        Write-Output "MTP printed a summary for $projectName but no 'total:' line matched. The counts"
        Write-Output "below are not measurements - do not read them as one. Check the summary format"
        Write-Output "against this script's parser before trusting any run."
        $failedProjects += $projectName
        $allResults += [PSCustomObject]@{
            Project = $projectName
            Passed = 0; Failed = 0; Skipped = 0; Total = 0
            ExitCode = $exitCode
        }
        continue
    }

    $allResults += [PSCustomObject]@{
        Project = $projectName
        Passed = $passed; Failed = $failed; Skipped = $skipped; Total = $total
        ExitCode = $exitCode
    }

    $totalPassed += $passed
    $totalFailed += $failed
    $totalSkipped += $skipped
    $totalTests += $total

    # MTP treats an assembly whose filter matches zero tests as a non-success exit (observed: 8,
    # where VSTest exited 0 - xunit/xunit#3077). $total is the ground truth for "did anything
    # actually fail" regardless of exit code here.
    # "The test host never ran" and "the filter matched nothing" are different facts, and only the
    # second one says anything about the tests. MTP prints a run summary whenever it actually ran an
    # assembly - even for zero matches, where it reports total: 0 - so the absence of a summary is
    # what separates them. Do not test for a particular message here: the wording differs by what is
    # missing (an unrestored project and an unbuilt one fail with different text on different
    # platforms), and a check keyed to one wording silently stops working when the other appears.
    #
    # This is the case that hid 108 tests in one assembly: the project was not in the solution, so
    # the build step never produced it, and a solution-wide dotnet test cannot mention an assembly it
    # was never given. A development machine hides it too - an earlier build leaves bin/ behind and
    # the project runs - so it only surfaces on a clean checkout, which is to say in CI.
    $neverRan = -not $sawSummary

    if ($total -eq 0) {
        $exempt = $hardwareOnlyProjects -contains $projectName
        if ($neverRan -and -not $exempt) {
            Write-Status "Result: THE TEST HOST NEVER RAN" "Red"
            Write-Output "dotnet test produced no run summary for $projectName, which is not the same as finding"
            Write-Output "no matching tests. The usual cause is that the project is missing from the solution, so"
            Write-Output "the build step never produced its assembly. Add it to the solution, or - if it is meant"
            Write-Output "to stay out - add it to `$hardwareOnlyProjects in this script and say why."
            $failedProjects += $projectName
        }
        elseif ($exempt) {
            $why = if ($neverRan) { "never built - outside the solution" } else { "filtered to nothing" }
            Write-Status "Result: no tests ran - $why (declared hardware-only)" "Yellow"
        }
        else {
            Write-Status "Result: NO TESTS RAN" "Red"
            Write-Output "This project is expected to contribute tests under the category filter and contributed none."
            Write-Output "Either something stopped running, or every test in it genuinely needs hardware or a"
            Write-Output "service - in which case add it to `$hardwareOnlyProjects in this script and say why."
            $failedProjects += $projectName
        }
    }
    elseif ($exitCode -eq 0) {
        Write-Status "Result: PASSED ($passed/$total)" "Green"
    }
    else {
        Write-Status "Result: FAILED ($failed failed, $passed passed, $skipped skipped)" "Red"
        $failedProjects += $projectName
    }
}

Write-Output ""
Write-Status "===================================" "Cyan"
Write-Status "Summary" "Cyan"
Write-Status "===================================" "Cyan"
Write-Output ""
Write-Output "Project                                  Passed  Failed  Skipped   Total"
Write-Output "------------------------------------------------------------------------"
foreach ($result in $allResults) {
    $line = "$($result.Project.PadRight(38)) $($result.Passed.ToString().PadLeft(7))  $($result.Failed.ToString().PadLeft(6))  $($result.Skipped.ToString().PadLeft(7))  $($result.Total.ToString().PadLeft(6))"
    $color = if ($result.Failed -gt 0 -or $failedProjects -contains $result.Project) { "Red" }
             elseif ($result.Total -eq 0) { "Yellow" }
             else { "Green" }
    Write-Status $line $color
}
Write-Output "------------------------------------------------------------------------"
Write-Output ""
Write-Output "  Total Tests:    $totalTests"
Write-Status "  Passed:         $totalPassed" "Green"
Write-Output "  Skipped:        $totalSkipped"
if ($totalFailed -gt 0) { Write-Status "  Failed:         $totalFailed" "Red" }
else                    { Write-Output "  Failed:         $totalFailed" }
Write-Output ""

if ($Filter) {
    Write-Status "NOTE: -Filter '$Filter' was applied. This run is not the gate." "Yellow"
    Write-Output ""
}

# Any failure fails the run. There is no acceptable pass rate below 100%: a test that is expected
# to fail is either a defect to fix or a category to exclude, and both are decisions to make
# explicitly rather than to absorb into a threshold.
if ($totalFailed -gt 0 -or $failedProjects.Count -gt 0) {
    Write-Status "OVERALL RESULT: FAILED" "Red"
    if ($failedProjects.Count -gt 0) {
        Write-Output "Failing project(s): $($failedProjects -join ', ')"
    }
    exit 1
}

Write-Status "OVERALL RESULT: PASSED ($totalPassed passed, $totalSkipped skipped)" "Green"
exit 0
