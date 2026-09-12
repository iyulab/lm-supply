using AwesomeAssertions;
using LMSupply.Captioner;
using LMSupply.Detector;
using LMSupply.Download;
using LMSupply.Embedder;
using LMSupply.Exceptions;
using LMSupply.ImageGenerator;
using LMSupply.Ocr;
using LMSupply.Reranker;
using LMSupply.Segmenter;
using LMSupply.Synthesizer;
using LMSupply.Transcriber;
using LMSupply.Translator;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// <c>DisableAutoDownload</c> against this machine's real model cache. Until 0.64.0 five of these modules
/// ignored the option and downloaded anyway, and until 0.65.0 the captioner, embedder, synthesizer and generator did not have it (the generator is covered by unit tests — an offline hit here would start llama-server). The unit tests count requests through a test transport; this
/// loads each module's default model the way a consumer does and checks the cache afterwards: an offline
/// load either opens a cached model or fails with <see cref="ModelNotFoundException"/>, and in both cases
/// leaves the cache exactly as it found it — nothing downloaded, nothing written. Needs the real cache,
/// so it is local only.
/// </summary>
[Trait("Category", "Functional")]
[Trait("Category", "LocalOnly")]
public sealed class DisableAutoDownloadRealCacheTests
{
    public static TheoryData<string> Modules =>
        ["transcriber", "translator", "reranker", "segmenter", "detector", "imagegenerator", "ocr", "captioner", "embedder", "synthesizer"];

    [Theory]
    [MemberData(nameof(Modules))]
    public async Task AnOfflineLoad_OpensTheCachedModelOrFails_AndLeavesTheCacheUntouched(string module)
    {
        var ct = TestContext.Current.CancellationToken;
        var cacheDir = CacheManager.GetDefaultCacheDirectory();
        var before = Snapshot(cacheDir);

        string outcome;
        try
        {
            await using var model = await LoadAsync(module, ct);
            outcome = "loaded from the cache";
        }
        catch (ModelNotFoundException ex)
        {
            outcome = $"not cached: {ex.Message}";
        }

        TestContext.Current.TestOutputHelper?.WriteLine($"{module}: {outcome}");

        // Name the entries that changed: a positional diff of hundreds of lines says nothing.
        var after = Snapshot(cacheDir);
        var added = after.Except(before).ToList();
        var removed = before.Except(after).ToList();
        (added.Count + removed.Count).Should().Be(0,
            $"an offline load ({module}: {outcome}) must not download or write anything, but it added or changed:\n" +
            $"{string.Join('\n', added)}\nand removed or changed:\n{string.Join('\n', removed)}");
    }

    private static async Task<IAsyncDisposable> LoadAsync(string module, CancellationToken ct) => module switch
    {
        "transcriber" => await LocalTranscriber.LoadAsync("default", new TranscriberOptions { DisableAutoDownload = true }, cancellationToken: ct),
        "translator" => await LocalTranslator.LoadAsync("default", new TranslatorOptions { DisableAutoDownload = true }, cancellationToken: ct),
        "reranker" => await LocalReranker.LoadAsync("default", new RerankerOptions { DisableAutoDownload = true }, cancellationToken: ct),
        "segmenter" => await LocalSegmenter.LoadAsync("default", new SegmenterOptions { DisableAutoDownload = true }, cancellationToken: ct),
        "detector" => await LocalDetector.LoadAsync("default", new DetectorOptions { DisableAutoDownload = true }, cancellationToken: ct),
        "imagegenerator" => await LocalImageGenerator.LoadAsync("default", new ImageGeneratorOptions { DisableAutoDownload = true }, cancellationToken: ct),
        "ocr" => await LocalOcr.LoadForLanguageAsync("en", new OcrOptions { DisableAutoDownload = true }, cancellationToken: ct),
        "captioner" => await LocalCaptioner.LoadAsync("default", new CaptionerOptions { DisableAutoDownload = true }, cancellationToken: ct),
        "embedder" => await LocalEmbedder.LoadAsync("default", new EmbedderOptions { DisableAutoDownload = true }, cancellationToken: ct),
        "synthesizer" => await LocalSynthesizer.LoadAsync("default", new SynthesizerOptions { DisableAutoDownload = true }, cancellationToken: ct),
        _ => throw new ArgumentOutOfRangeException(nameof(module), module, null),
    };

    // Every file under the model directories, with size and write time: a download adds files, a manifest
    // rewrite changes a write time, and a miss that creates an empty directory adds a directory entry.
    private static List<string> Snapshot(string cacheDir)
    {
        if (!Directory.Exists(cacheDir))
            return [];

        var entries = new List<string>();
        foreach (var modelDir in Directory.GetDirectories(cacheDir, "models--*"))
        {
            entries.Add(Path.GetFileName(modelDir) + "/");
            foreach (var file in Directory.EnumerateFiles(modelDir, "*", SearchOption.AllDirectories))
            {
                var info = new FileInfo(file);
                entries.Add($"{Path.GetRelativePath(cacheDir, file)} {info.Length} {info.LastWriteTimeUtc.Ticks}");
            }
        }

        return entries;
    }
}
