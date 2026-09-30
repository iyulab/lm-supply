using AwesomeAssertions;
using Xunit;

namespace LMSupply.Core.Tests.Inference;

/// <summary>
/// Every model type takes its session's threads and log level from <c>LMSupplyOptionsBase</c> through one method,
/// <c>SessionOptionsExtensions.ApplyCommonOptions</c>. Until 0.97.0 each model type copied the rule, and the copies
/// drifted: the Embedder fixed intra-op threads at the logical core count and ignored <c>ThreadCount</c>, OCR and the
/// Captioner ignored it too, the Reranker used a rule of its own, Parakeet ignored the log level, and nothing turned off
/// the thread pool's spin-wait — so a short embedding call kept several cores busy for seconds afterwards.
/// </summary>
public sealed class SessionThreadingConventionTests
{
    private static readonly string[] SessionSettings = ["IntraOpNumThreads", "InterOpNumThreads", "LogSeverityLevel ="];

    [Fact]
    public void ModelSources_SetThreadsAndLogLevel_OnlyThroughApplyCommonOptions()
    {
        var root = new DirectoryInfo(AppContext.BaseDirectory);
        while (root is not null && !File.Exists(Path.Combine(root.FullName, "lm-supply.slnx")))
            root = root.Parent;
        root.Should().NotBeNull("the test runs under the repository");

        var allowed = new[]
        {
            Path.Combine("LMSupply.Core", "Inference", "SessionOptionsExtensions.cs"),
            // The provider probe builds a throwaway session with a fixed error log level; it never runs a model.
            Path.Combine("LMSupply.Core", "Inference", "OnnxSessionFactory.cs"),
        };

        var offenders = Directory.EnumerateFiles(Path.Combine(root!.FullName, "src"), "*.cs", SearchOption.AllDirectories)
            .Where(f => !f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal))
            .Where(f => !allowed.Any(a => f.EndsWith(a, StringComparison.Ordinal)))
            .SelectMany(f => File.ReadLines(f)
                .Select((line, i) => (File: f, Line: i + 1, Text: line.Trim()))
                .Where(l => !l.Text.StartsWith("//", StringComparison.Ordinal)
                            && SessionSettings.Any(s => l.Text.Contains(s, StringComparison.Ordinal))))
            .Select(l => $"{Path.GetRelativePath(root.FullName, l.File)}:{l.Line}: {l.Text}")
            .ToList();

        offenders.Should().BeEmpty("a model type applies ThreadCount and LogLevel with sessionOptions.ApplyCommonOptions(options)");
    }

    [Fact]
    public void TheOnlyThreadSetting_IsTheSharedOne()
    {
        // Positive control for the scan above: it must find the settings where they are.
        var root = new DirectoryInfo(AppContext.BaseDirectory);
        while (root is not null && !File.Exists(Path.Combine(root.FullName, "lm-supply.slnx")))
            root = root.Parent;

        var shared = File.ReadAllText(Path.Combine(root!.FullName, "src", "LMSupply.Core", "Inference", "SessionOptionsExtensions.cs"));
        shared.Should().Contain("IntraOpNumThreads").And.Contain("allow_spinning");
    }
}
