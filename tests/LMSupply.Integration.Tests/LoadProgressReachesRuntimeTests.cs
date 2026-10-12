using AwesomeAssertions;
using LMSupply.Captioner;
using LMSupply.Detector;
using LMSupply.Embedder;
using LMSupply.Exceptions;
using LMSupply.Generator;
using LMSupply.ImageGenerator;
using LMSupply.Ocr;
using LMSupply.Reranker;
using LMSupply.Segmenter;
using LMSupply.Synthesizer;
using LMSupply.Transcriber;
using LMSupply.Translator;

namespace LMSupply.Integration.Tests;

/// <summary>
/// Every ONNX domain's <c>LoadAsync(…, progress, …)</c> hands its progress to session creation too, so the runtime step
/// of a first load is visible: a download reports <see cref="DownloadKind.Runtime"/> bytes, and a runtime already on disk
/// reports one <see cref="DownloadKind.Runtime"/> <see cref="DownloadPhase.Complete"/>. Either way at least one runtime
/// report arrives. <see cref="LoadProgressReachesDownloadTests"/> covers the model half from an empty cache; this loads
/// from the machine's real cache with downloads off, so it is local only and skips a domain whose model is not cached.
/// </summary>
[Trait("Category", "Integration")]
[Trait("Category", "LocalOnly")]
public sealed class LoadProgressReachesRuntimeTests
{
    public static TheoryData<string> Domains =>
    [
        "Captioner", "Detector", "Embedder", "ImageGenerator", "Ocr", "Reranker",
        "Segmenter", "InteractiveSegmenter", "Synthesizer", "Transcriber", "Translator",
    ];

    [Theory]
    [MemberData(nameof(Domains))]
    public async Task LoadReportsTheRuntimeStep(string domain)
    {
        var ct = TestContext.Current.CancellationToken;
        var reports = new List<DownloadProgress>();
        var progress = new SyncProgress(p => { lock (reports) reports.Add(p); });

        try
        {
            await using var model = await LoadAsync(domain, progress, ct);
        }
        catch (ModelNotFoundException ex)
        {
            Assert.Skip($"{domain}: the default model is not cached here ({ex.Message}).");
        }

        List<DownloadProgress> seen;
        lock (reports) seen = [.. reports];
        seen.Should().Contain(r => r.Kind == DownloadKind.Runtime,
            $"{domain}.LoadAsync must forward its progress to session creation; saw: " +
            string.Join(" | ", seen.Take(8).Select(r => $"{r.Kind}:{r.FileName}:{r.Phase}")));
    }

    public static TheoryData<string> DomainsAndTheGgufGenerator => [.. Domains, "GgufGenerator"];

    // Every model here is already on disk, so what is left of the load is bringing the model into memory. That stage
    // is reported as Loading, with nothing counted; it does not borrow a download phase or invent byte counts, and a
    // file that was reported Complete is not completed again.
    [Theory]
    [MemberData(nameof(DomainsAndTheGgufGenerator))]
    public async Task AWarmLoadReportsLoading_AndNothingItDidNotMeasure(string domain)
    {
        var ct = TestContext.Current.CancellationToken;
        var reports = new List<DownloadProgress>();
        var progress = new SyncProgress(p => { lock (reports) reports.Add(p); });

        try
        {
            await using var model = await LoadAsync(domain, progress, ct);
        }
        catch (ModelNotFoundException ex)
        {
            Assert.Skip($"{domain}: the model is not cached here ({ex.Message}).");
        }

        List<DownloadProgress> seen;
        lock (reports) seen = [.. reports];
        var trace = string.Join(" | ", seen.Select(r => $"{r.Kind}:{r.FileName}:{r.Phase}:{r.BytesDownloaded}/{r.TotalBytes}"));
        TestContext.Current.TestOutputHelper?.WriteLine($"{domain}: {trace}");
        var modelReports = seen.Where(r => r.Kind == DownloadKind.Model).ToList();

        var loading = modelReports.Where(r => r.Phase == DownloadPhase.Loading).ToList();
        loading.Should().NotBeEmpty($"{domain}: the load brings a model into memory; saw: {trace}");
        loading.Should().OnlyContain(r => r.IsIndeterminate && r.BytesDownloaded == 0 && r.TotalBytes == 0,
            $"nothing is counted while a model loads; saw: {trace}");
        loading.Select(r => r.FileName).Should().OnlyHaveUniqueItems($"a file is loaded once; saw: {trace}");

        modelReports.Should().NotContain(r => r.Phase == DownloadPhase.Downloading || r.Phase == DownloadPhase.Extracting,
            $"the files were on disk and none of them is an archive; saw: {trace}");
        modelReports.Where(r => r.Phase == DownloadPhase.Complete).Select(r => r.FileName)
            .Should().OnlyHaveUniqueItems($"a file is completed once; saw: {trace}");
        foreach (var file in loading.Select(r => r.FileName))
        {
            var loadingAt = seen.FindIndex(r => r.Phase == DownloadPhase.Loading && r.FileName == file);
            seen.Skip(loadingAt + 1).Should().NotContain(r => r.Kind == DownloadKind.Model && r.FileName == file,
                $"Loading is the last report of a file — the load is over when LoadAsync returns; saw: {trace}");
        }
        seen.Where(r => r.Phase != DownloadPhase.Complete && r.TotalBytes > 0).Should()
            .OnlyContain(r => r.Phase == DownloadPhase.Downloading || r.Phase == DownloadPhase.Extracting || r.Phase == DownloadPhase.Verifying,
                $"only a step that counts bytes carries a total; saw: {trace}");
    }

    private static Task<IAsyncDisposable> LoadAsync(string domain, IProgress<DownloadProgress> progress, CancellationToken ct) =>
        domain switch
        {
            "GgufGenerator" => Box(LocalGenerator.LoadAsync(CachedGgufRepository, new GeneratorOptions { DisableAutoDownload = true }, progress, ct)),
            "Captioner" => Box(LocalCaptioner.LoadAsync("default", new CaptionerOptions { DisableAutoDownload = true }, progress, ct)),
            "Detector" => Box(LocalDetector.LoadAsync("default", new DetectorOptions { DisableAutoDownload = true }, progress, ct)),
            "Embedder" => Box(LocalEmbedder.LoadAsync("default", new EmbedderOptions { DisableAutoDownload = true }, progress, ct)),
            "ImageGenerator" => Box(LocalImageGenerator.LoadAsync("default", new ImageGeneratorOptions { DisableAutoDownload = true }, progress, ct)),
            "Ocr" => Box(LocalOcr.LoadForLanguageAsync("en", new OcrOptions { DisableAutoDownload = true }, progress, ct)),
            "Reranker" => Box(LocalReranker.LoadAsync("default", new RerankerOptions { DisableAutoDownload = true }, progress, ct)),
            "Segmenter" => Box(LocalSegmenter.LoadAsync("default", new SegmenterOptions { DisableAutoDownload = true }, progress, ct)),
            "InteractiveSegmenter" => Box(LocalSegmenter.LoadInteractiveAsync("interactive", new SegmenterOptions { DisableAutoDownload = true }, progress, ct)),
            "Synthesizer" => Box(LocalSynthesizer.LoadAsync("default", new SynthesizerOptions { DisableAutoDownload = true }, progress, ct)),
            "Transcriber" => Box(LocalTranscriber.LoadAsync("fast", new TranscriberOptions { DisableAutoDownload = true }, progress, ct)),
            "Translator" => Box(LocalTranslator.LoadAsync("default", new TranslatorOptions { DisableAutoDownload = true }, progress, ct)),
            _ => throw new ArgumentOutOfRangeException(nameof(domain)),
        };

    // A small GGUF repository; the test skips where it is not cached.
    private const string CachedGgufRepository = "unsloth/Qwen3.5-2B-GGUF";

    private static async Task<IAsyncDisposable> Box<T>(Task<T> load) where T : IAsyncDisposable => await load;

    private sealed class SyncProgress(Action<DownloadProgress> onReport) : IProgress<DownloadProgress>
    {
        public void Report(DownloadProgress value) => onReport(value);
    }
}
