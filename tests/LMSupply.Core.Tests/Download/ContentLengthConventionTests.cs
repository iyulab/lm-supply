using System.Text.RegularExpressions;
using AwesomeAssertions;
using Xunit;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// A model file in the hub cache is a snapshot entry that is a symbolic link to its blob, and
/// <see cref="FileInfo.Length"/> of a link is the link's own length (0 on Windows, the length of the target path
/// elsewhere), never the model's. Sizing a model by it read a 1.3 GB GGUF as 0 bytes on Windows and divided by zero
/// in the GPU offload estimate; on other systems it sized the model at a few dozen bytes. Model sizes are read with
/// <c>CacheManager.TryGetContentLength</c> (or <c>GetContentLength</c>), which follows the link.
/// </summary>
public sealed partial class ContentLengthConventionTests
{
    [GeneratedRegex(@"new FileInfo\([^)]*\)\.Length|(?:EnumerateFiles|GetFiles)\(.*\)\.Sum\(\s*\w+\s*=>\s*\w+\.Length\s*\)")]
    private static partial Regex LinkLengthRead();

    // Files that read a length on purpose: the cache itself (a link's own entry, a ref file, a partial download).
    private static readonly string[] Allowed =
    [
        Path.Combine("LMSupply.Core", "Download", "CacheManager.cs"),
        Path.Combine("LMSupply.Core", "Download", "HubCache.cs"),
        Path.Combine("LMSupply.Core", "Download", "ResumableFileDownload.cs"),
    ];

    [Fact]
    public void ModelSources_ReadFileLengths_ThroughTheCacheManager()
    {
        var root = RepositoryRoot();

        var offenders = Directory.EnumerateFiles(Path.Combine(root, "src"), "*.cs", SearchOption.AllDirectories)
            .Where(f => !f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal))
            .Where(f => !Allowed.Any(a => f.EndsWith(a, StringComparison.Ordinal)))
            .SelectMany(f => File.ReadLines(f)
                .Select((line, i) => (File: f, Line: i + 1, Text: line.Trim()))
                .Where(l => !l.Text.StartsWith("//", StringComparison.Ordinal) && LinkLengthRead().IsMatch(l.Text)))
            .Select(l => $"{Path.GetRelativePath(root, l.File)}:{l.Line}: {l.Text}")
            .ToList();

        offenders.Should().BeEmpty("a model file may be a hub-cache link: read its size with CacheManager.TryGetContentLength");
    }

    [Fact]
    public void TheScan_FindsTheCachesOwnReads()
    {
        // Positive control for the scan above: it must match where such reads are.
        var cacheManager = File.ReadAllText(Path.Combine(RepositoryRoot(), "src", "LMSupply.Core", "Download", "CacheManager.cs"));
        LinkLengthRead().IsMatch(cacheManager).Should().BeTrue();
    }

    private static string RepositoryRoot()
    {
        var root = new DirectoryInfo(AppContext.BaseDirectory);
        while (root is not null && !File.Exists(Path.Combine(root.FullName, "lm-supply.slnx")))
            root = root.Parent;
        root.Should().NotBeNull("the test runs under the repository");
        return root!.FullName;
    }
}
