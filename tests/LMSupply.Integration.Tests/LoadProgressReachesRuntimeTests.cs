using AwesomeAssertions;
using LMSupply.Captioner;
using LMSupply.Detector;
using LMSupply.Embedder;
using LMSupply.Exceptions;
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

    private static Task<IAsyncDisposable> LoadAsync(string domain, IProgress<DownloadProgress> progress, CancellationToken ct) =>
        domain switch
        {
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

    private static async Task<IAsyncDisposable> Box<T>(Task<T> load) where T : IAsyncDisposable => await load;

    private sealed class SyncProgress(Action<DownloadProgress> onReport) : IProgress<DownloadProgress>
    {
        public void Report(DownloadProgress value) => onReport(value);
    }
}
