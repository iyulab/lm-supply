using Iyu.Conventions.Testing;
using Xunit;

namespace LMSupply.Integration.Tests;

/// <summary>
/// Operational text - every <c>[LoggerMessage]</c> template and every exception message - is ASCII. Operators grep it,
/// paste it into issues and search it in log pipelines whose tokenizers split on Latin word boundaries; a dash or an
/// arrow outside ASCII is as opaque there as a Korean word. The scan is <c>Iyu.Conventions.Testing</c>'s, shared with
/// the other repositories, over the same assemblies the options roster scans (every LMSupply library assembly).
/// </summary>
public class OperationalLanguageConventionTests
{
    private static readonly Lazy<OperationalLanguageReport> Result = new(() =>
        OperationalLanguage.Scan(OptionsReachabilityRosterTests.LibraryAssemblies(), OperationalLanguage.NonAscii));

    [Fact]
    public void LogTemplatesAndExceptionMessages_AreAscii()
    {
        var findings = Result.Value.Findings;
        Assert.True(findings.Count == 0,
            "Non-ASCII operational text:\n" + string.Join("\n", findings.Select(f => $"  [{f.Kind}] {f.Location}: {f.Text}")));
    }

    // Positive control: the scan must see the operational text it exists to judge. LMSupply logs through ILogger calls,
    // not [LoggerMessage] templates, so the exception messages are what it reads here.
    [Fact]
    public void Scan_SeesExceptionMessages() =>
        Assert.True(Result.Value.ExceptionLiteralsRead > 50, $"exception messages read: {Result.Value.ExceptionLiteralsRead}");
}
