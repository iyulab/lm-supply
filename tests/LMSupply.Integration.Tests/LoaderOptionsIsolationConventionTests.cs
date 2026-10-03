using System.Text.RegularExpressions;
using Xunit;

namespace LMSupply.Integration.Tests;

/// <summary>
/// The <c>Local*</c> entry points resolve a <c>:variant</c> qualifier, a model id or an alias into their options. Done
/// on the caller's instance, that leaks into the caller's next load — a <c>"quality:fp16"</c> load left
/// <c>QuantizationHint = "fp16"</c> on an options object later used for another model. Every entry point therefore
/// works on <c>options?.Clone() ?? new …()</c>; this scan fails on the shape that mutates the caller's instance.
/// </summary>
public class LoaderOptionsIsolationConventionTests
{
    private static readonly Regex MutatesCallerOptions = new(@"\boptions\s*\?\?=\s*new\b", RegexOptions.Compiled);

    [Fact]
    public void LocalEntryPoints_WorkOnACopyOfTheCallersOptions()
    {
        var root = RepositoryRoot();
        var files = Directory.EnumerateFiles(Path.Combine(root, "src"), "Local*.cs", SearchOption.AllDirectories)
            .Where(f => !f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal))
            .ToList();

        Assert.NotEmpty(files);

        var offenders = files
            .SelectMany(file => File.ReadLines(file)
                .Select((line, index) => (line, index))
                .Where(x => MutatesCallerOptions.IsMatch(x.line))
                .Select(x => $"{Path.GetRelativePath(root, file)}:{x.index + 1}: {x.line.Trim()}"))
            .ToList();

        Assert.True(offenders.Count == 0,
            "Use `options = options?.Clone() ?? new …()` so the entry point never changes the caller's options:\n"
            + string.Join("\n", offenders));
    }

    [Fact]
    public void TheScan_CatchesTheMutatingShape()
    {
        // Positive control: the pattern this convention forbids is the one the regex matches.
        Assert.Matches(MutatesCallerOptions, "        options ??= new CaptionerOptions();");
        Assert.DoesNotMatch(MutatesCallerOptions, "        options = options?.Clone() ?? new CaptionerOptions();");
    }

    private static string RepositoryRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "lm-supply.slnx")))
            dir = dir.Parent;
        return dir?.FullName ?? throw new InvalidOperationException("lm-supply.slnx not found above the test output directory");
    }
}
