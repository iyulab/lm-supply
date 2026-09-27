using System.Collections;
using System.Reflection;
using AwesomeAssertions;
using LMSupply.Core.Download;
using LMSupply.Generator.Internal.Llama;

namespace LMSupply.Integration.Tests;

/// <summary>
/// Every registry alias points at a repository that exists and lists the files the entry pins. A dead entry fails only when
/// someone loads it — the Translator's en-ko alias named a repository that never existed, and MobileSAM named a GitHub
/// project instead of a Hugging Face repository; both shipped for many releases because nothing loaded them.
/// </summary>
/// <remarks>
/// One listing request per distinct repository (about fifty), against the anonymous Hugging Face API quota the machine
/// shares with everything else on it — hence Integration, not CI.
/// </remarks>
[Trait("Category", "Integration")]
public sealed class RegistryRepositoriesExistTests
{
    // One type per domain assembly, so every registry assembly is loaded before the reflection scan.
    private static readonly Assembly[] DomainAssemblies =
    [
        typeof(LMSupply.Embedder.LocalEmbedder).Assembly,
        typeof(LMSupply.Reranker.LocalReranker).Assembly,
        typeof(LMSupply.Generator.LocalGenerator).Assembly,
        typeof(LMSupply.Translator.LocalTranslator).Assembly,
        typeof(LMSupply.Transcriber.LocalTranscriber).Assembly,
        typeof(LMSupply.Synthesizer.LocalSynthesizer).Assembly,
        typeof(LMSupply.Captioner.LocalCaptioner).Assembly,
        typeof(LMSupply.Ocr.LocalOcr).Assembly,
        typeof(LMSupply.Detector.LocalDetector).Assembly,
        typeof(LMSupply.Segmenter.LocalSegmenter).Assembly,
        typeof(LMSupply.ImageGenerator.LocalImageGenerator).Assembly,
    ];

    [Fact]
    public async Task EveryRegistryEntry_NamesAnExistingRepository_ThatListsItsPinnedFiles()
    {
        var entries = RegistryEntries().ToList();
        entries.Should().HaveCountGreaterThan(40, "the scan must find every domain's registry, or a pass proves nothing");

        using var discovery = new ModelDiscoveryService(Path.Combine(Path.GetTempPath(), "lmsupply-registry-scan"));
        var problems = new List<string>();
        foreach (var repo in entries.GroupBy(e => e.Repo, StringComparer.Ordinal))
        {
            IReadOnlyList<RepoFile> files;
            try
            {
                files = await discovery.ListRepositoryFilesAsync(repo.Key, cancellationToken: TestContext.Current.CancellationToken);
            }
            catch (Exception ex) when (ex is LMSupply.Exceptions.ModelNotFoundException or HttpRequestException or UnauthorizedAccessException)
            {
                problems.Add($"{repo.Key}: not listable ({ex.GetType().Name}) — used by {string.Join(", ", repo.Select(e => e.Source).Distinct())}");
                continue;
            }

            var paths = files.Where(f => f.IsFile).Select(f => f.Path).ToHashSet(StringComparer.Ordinal);
            foreach (var entry in repo)
            foreach (var file in entry.PinnedFiles)
            {
                if (!paths.Contains(file) && !paths.Any(p => p.EndsWith("/" + file, StringComparison.Ordinal)))
                    problems.Add($"{repo.Key}: no file '{file}' — pinned by {entry.Source}");
            }
        }

        problems.Should().BeEmpty("every alias must be loadable:" + Environment.NewLine + string.Join(Environment.NewLine, problems));
    }

    private sealed record Entry(string Repo, string Source, IReadOnlyList<string> PinnedFiles);

    private static IEnumerable<Entry> RegistryEntries()
    {
        // Every public static «All» list on a Default*Models type in the domain assemblies.
        foreach (var type in DomainAssemblies.SelectMany(a => a.GetTypes())
                     .Where(t => t.IsAbstract && t.IsSealed && t.Name.StartsWith("Default", StringComparison.Ordinal) && t.Name.EndsWith("Models", StringComparison.Ordinal)))
        {
            if (type.GetProperty("All", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)?.GetValue(null) is not IEnumerable all)
                continue;
            foreach (var model in all)
            {
                // «org/name:variant» — the qualifier picks a file inside the repository (LMSupplyOptionsBase.SplitQualifier).
                var repo = (StringProperty(model, "Id") ?? StringProperty(model, "RepoId"))?.Split(':')[0];
                if (repo is null || repo.Count(c => c == '/') != 1 || Path.IsPathRooted(repo))
                    continue;
                // An interactive (encoder + decoder) segmenter entry keeps the semantic models' OnnxFile default; it loads only
                // its EncoderFile and DecoderFile.
                var interactive = model.GetType().GetProperty("IsInteractive")?.GetValue(model) is true;
                var pinned = model.GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance)
                    .Where(p => p.PropertyType == typeof(string) && p.Name.EndsWith("File", StringComparison.Ordinal))
                    .Where(p => !(interactive && p.Name == "OnnxFile"))
                    .Select(p => (string?)p.GetValue(model))
                    .Where(v => !string.IsNullOrEmpty(v))
                    .Select(v => v!)
                    .ToList();
                yield return new Entry(repo, $"{type.FullName}[{StringProperty(model, "AliasName") ?? repo}]", pinned);
            }
        }

        foreach (var gguf in GgufModelRegistry.GetAllModels())
            yield return new Entry(gguf.RepoId, $"GgufModelRegistry[{gguf.RepoId}]", [gguf.DefaultFile]);
    }

    private static string? StringProperty(object model, string name) =>
        model.GetType().GetProperty(name, BindingFlags.Public | BindingFlags.Instance)?.GetValue(model) as string;
}
