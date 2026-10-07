using AwesomeAssertions;
using LMSupply.Detector;
using LMSupply.Reranker;
using LMSupply.Segmenter;
using LMSupply.Synthesizer;
using LMSupply.Transcriber;
using LMSupply.Translator;

namespace LMSupply.Integration.Tests;

/// <summary>
/// Every <c>Local*.LoadAsync(…, progress, …)</c> hands its progress to the download it makes. Each case loads into an
/// empty cache, so a download must happen, and stops at the first byte report. <see cref="DownloadStartTests"/> reads the
/// shared cache and counts a cached model as success, so it cannot tell a dropped progress from a warm cache.
/// </summary>
[Trait("Category", "Integration")]
public sealed class LoadProgressReachesDownloadTests : IDisposable
{
    private readonly string _cache = Path.Combine(Path.GetTempPath(), "lmsupply-load-progress-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_cache))
                Directory.Delete(_cache, recursive: true);
        }
        catch (IOException)
        {
            // A cancelled download may still hold a file for a moment; the temp directory is not worth failing over.
        }
    }

    public static TheoryData<string> Domains => ["Detector", "Reranker", "Segmenter", "InteractiveSegmenter", "Synthesizer", "Translator", "Transcriber"];

    [Theory]
    [MemberData(nameof(Domains))]
    public async Task LoadReportsItsDownload(string domain)
    {
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        cts.CancelAfter(TimeSpan.FromSeconds(60));
        var reported = false;
        var seen = new List<string>();
        var progress = new SyncProgress(p =>
        {
            lock (seen) { if (seen.Count < 8) seen.Add($"{p.FileName}:{p.Phase}:{p.BytesDownloaded}/{p.TotalBytes}"); }
            if (p.BytesDownloaded > 0)
            {
                reported = true;
                cts.Cancel();
            }
        });

        try
        {
            await using var model = await LoadAsync(domain, progress, cts.Token);
        }
        catch (Exception ex) when (ex is OperationCanceledException || ex.InnerException is OperationCanceledException)
        {
            // Cancelled at the first byte report, or by the 60 s limit when no report ever came.
        }

        reported.Should().BeTrue($"{domain}.LoadAsync into an empty cache must report the download through its progress; seen: " + string.Join(" | ", seen));
    }

    private Task<IAsyncDisposable> LoadAsync(string domain, IProgress<DownloadProgress> progress, CancellationToken ct) =>
        domain switch
        {
            "Detector" => Box(LocalDetector.LoadAsync("default", new DetectorOptions { CacheDirectory = _cache }, progress, ct)),
            "Reranker" => Box(LocalReranker.LoadAsync("default", new RerankerOptions { CacheDirectory = _cache }, progress, ct)),
            "Segmenter" => Box(LocalSegmenter.LoadAsync("default", new SegmenterOptions { CacheDirectory = _cache }, progress, ct)),
            "InteractiveSegmenter" => Box(LocalSegmenter.LoadInteractiveAsync("interactive", new SegmenterOptions { CacheDirectory = _cache }, progress, ct)),
            "Synthesizer" => Box(LocalSynthesizer.LoadAsync("default", new SynthesizerOptions { CacheDirectory = _cache }, progress, ct)),
            "Translator" => Box(LocalTranslator.LoadAsync("default", new TranslatorOptions { CacheDirectory = _cache }, progress, ct)),
            "Transcriber" => Box(LocalTranscriber.LoadAsync("fast", new TranscriberOptions { CacheDirectory = _cache }, progress, ct)),
            _ => throw new ArgumentOutOfRangeException(nameof(domain)),
        };

    private static async Task<IAsyncDisposable> Box<T>(Task<T> load) where T : IAsyncDisposable => await load;

    // Progress<T> posts to the thread pool; the cancellation must follow the report, not race it.
    private sealed class SyncProgress(Action<DownloadProgress> onReport) : IProgress<DownloadProgress>
    {
        public void Report(DownloadProgress value) => onReport(value);
    }
}
