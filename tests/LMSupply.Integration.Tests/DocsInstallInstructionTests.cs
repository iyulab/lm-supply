using System.Text.RegularExpressions;
using AwesomeAssertions;
using Xunit;

namespace LMSupply.Integration.Tests;

/// <summary>
/// What the documentation tells a reader to install must be something to install. <see cref="DocsSnippetRosterTests"/>
/// checks the C# blocks; an install line is shell, and it is the first thing a reader copies. Install lines for an
/// execution provider that was removed, or for a package that never existed, stayed in the package READMEs shown on
/// nuget.org long after the code had moved on.
/// </summary>
public partial class DocsInstallInstructionTests
{
    /// <summary>Packages outside this repository that a document may tell a reader to add.</summary>
    private static readonly HashSet<string> KnownExternal = new(StringComparer.OrdinalIgnoreCase)
    {
    };

    [Fact]
    public void EveryPackageADocumentSaysToAdd_IsOneOfOurs()
    {
        var root = RepositoryRoot();
        var ours = Directory.EnumerateDirectories(Path.Combine(root, "src"))
            .Select(Path.GetFileName)
            .Where(name => File.Exists(Path.Combine(root, "src", name!, name + ".csproj")))
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        var findings = new List<string>();
        var installs = 0;
        foreach (var file in DocumentFiles(root))
        {
            var lines = File.ReadAllLines(file);
            for (var i = 0; i < lines.Length; i++)
            {
                foreach (Match match in AddPackage().Matches(lines[i]))
                {
                    installs++;
                    var package = match.Groups["id"].Value;
                    if (!ours.Contains(package) && !KnownExternal.Contains(package))
                    {
                        findings.Add($"{Path.GetRelativePath(root, file)}:{i + 1}: {package}");
                    }
                }
            }
        }

        installs.Should().BeGreaterThan(0, "the package READMEs carry install lines; finding none means the scan is broken");
        findings.Should().BeEmpty(
            "LMSupply provisions ONNX Runtime and llama-server itself, so a document only ever adds LMSupply packages; " +
            "anything else needs a reason in KnownExternal");
    }

    private static IEnumerable<string> DocumentFiles(string root) =>
        new[] { Path.Combine(root, "README.md") }
            .Concat(Directory.EnumerateFiles(Path.Combine(root, "docs"), "*.md"))
            .Concat(Directory.EnumerateDirectories(Path.Combine(root, "src")).Select(d => Path.Combine(d, "README.md")))
            .Where(File.Exists)
            .Order(StringComparer.Ordinal);

    [GeneratedRegex(@"dotnet add package (?<id>[A-Za-z0-9_.\-]+)")]
    private static partial Regex AddPackage();

    private static string RepositoryRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "lm-supply.slnx")))
            dir = dir.Parent;
        return dir?.FullName ?? throw new InvalidOperationException("lm-supply.slnx not found above the test output directory");
    }
}
