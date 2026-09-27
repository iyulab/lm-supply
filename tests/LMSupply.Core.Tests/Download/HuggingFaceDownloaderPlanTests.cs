using System.Net;
using LMSupply.Core.Download;
using LMSupply.Download;
using LMSupply.Exceptions;

namespace LMSupply.Core.Tests.Download;

/// <summary>
/// A download plan answers what the matching download fetches. The invariant each test holds is not a figure but an
/// equality: the plan's files and bytes are the files and bytes the download then writes into an empty cache — a
/// plan that is only "about right" is the defect it exists to remove (a size that stopped matching the download).
/// </summary>
public sealed class HuggingFaceDownloaderPlanTests : IDisposable
{
    private const string Repo = "acme/whisper-like";
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-plan-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    // An encoder-decoder export with a full-precision and an int8 pair, laid out as Whisper repositories are.
    private static readonly Dictionary<string, int> Files = new(StringComparer.Ordinal)
    {
        ["onnx/encoder_model.onnx"] = 4000,
        ["onnx/encoder_model_int8.onnx"] = 1000,
        ["onnx/decoder_model_merged.onnx"] = 6000,
        ["onnx/decoder_model_merged_int8.onnx"] = 1500,
        ["config.json"] = 30,
        ["generation_config.json"] = 20,
        ["preprocessor_config.json"] = 25,
        ["tokenizer.json"] = 90,
        ["tokenizer_config.json"] = 40,
        ["root_only.bin"] = 7,
    };

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    public static TheoryData<string> Quantizations => new() { "int8", "fp32" };

    [Theory]
    [MemberData(nameof(Quantizations))]
    public async Task DiscoveryPlan_IsExactlyWhatTheDownloadWrites(string hint)
    {
        var preferences = new ModelPreferences
        {
            PreferredSubfolder = "onnx",
            QuantizationPriority = ModelPreferences.ForQuantizationHint(hint).QuantizationPriority,
            RequireMatchedQuantization = true
        };
        var hub = new Hub();
        using var downloader = new HuggingFaceDownloader(_cacheDir, hub);

        var plan = await downloader.PlanWithDiscoveryAsync(Repo, preferences, cancellationToken: Ct);
        var (dir, _) = await downloader.DownloadWithDiscoveryAsync(Repo, preferences, cancellationToken: Ct);

        Assert.Equal(WrittenFiles(dir), plan.Files.Select(f => f.Path).Order(StringComparer.Ordinal));
        Assert.Equal(WrittenBytes(dir), plan.TotalBytes);
        // The quantization is the one asked for — the plan does not fall back to the full-precision size.
        var onnx = plan.Files.Where(f => f.Path.EndsWith(".onnx", StringComparison.Ordinal)).Select(f => f.Path).ToList();
        Assert.Equal(2, onnx.Count);
        Assert.All(onnx, f => Assert.Equal(hint == "int8", f.Contains("_int8", StringComparison.Ordinal)));
        // One listing serves both: the plan's request is the one the download reuses.
        Assert.Equal(1, hub.ListingRequests);
    }

    [Fact]
    public async Task ExplicitPlan_TakesTheSubfolderOrTheRootFallback_AndSkipsAFileTheRepositoryLacks()
    {
        using var downloader = new HuggingFaceDownloader(_cacheDir, new Hub());
        string[] files = ["encoder_model_int8.onnx", "tokenizer.json", "vocab.txt"];

        var plan = await downloader.PlanModelAsync(Repo, files, subfolder: "onnx", cancellationToken: Ct);
        var dir = await downloader.DownloadModelAsync(Repo, files, subfolder: "onnx", cancellationToken: Ct);

        Assert.Equal(["onnx/encoder_model_int8.onnx", "tokenizer.json"], plan.Files.Select(f => f.Path));
        Assert.Equal(1000 + 90, plan.TotalBytes);
        Assert.Equal(plan.TotalBytes, Directory.GetFiles(dir).Where(IsPayload).Sum(f => new FileInfo(f).Length));
    }

    [Fact]
    public async Task ExplicitPlan_ThrowsForAMissingGraph_AsTheDownloadDoes()
    {
        using var downloader = new HuggingFaceDownloader(_cacheDir, new Hub());

        await Assert.ThrowsAsync<ModelDownloadException>(() =>
            downloader.PlanModelAsync(Repo, ["missing.onnx"], cancellationToken: Ct));
        await Assert.ThrowsAsync<ModelDownloadException>(() =>
            downloader.DownloadModelAsync(Repo, ["missing.onnx"], cancellationToken: Ct));
    }

    // Offline, the plan comes from what an earlier online run cached — no request, same answer.
    [Fact]
    public async Task Plan_WithLocalFilesOnly_AnswersFromTheCacheWithoutARequest()
    {
        using (var online = new HuggingFaceDownloader(_cacheDir, new Hub()))
            await online.DownloadWithDiscoveryAsync(Repo, cancellationToken: Ct);
        var offlineHub = new Hub();
        using var offline = new HuggingFaceDownloader(_cacheDir, offlineHub, localFilesOnly: true);

        var plan = await offline.PlanWithDiscoveryAsync(Repo, cancellationToken: Ct);

        Assert.NotEmpty(plan.Files);
        Assert.Equal(0, offlineHub.Requests);
    }

    private static bool IsPayload(string path) => !Path.GetFileName(path).StartsWith('.');

    private static IEnumerable<string> WrittenFiles(string dir) =>
        Directory.GetFiles(dir, "*", SearchOption.AllDirectories)
            .Where(IsPayload)
            .Select(f => Path.GetRelativePath(dir, f).Replace(Path.DirectorySeparatorChar, '/'))
            .Order(StringComparer.Ordinal);

    private static long WrittenBytes(string dir) =>
        Directory.GetFiles(dir, "*", SearchOption.AllDirectories).Where(IsPayload).Sum(f => new FileInfo(f).Length);

    /// <summary>Serves the listing and every listed file at its listed length; counts requests.</summary>
    private sealed class Hub : HttpMessageHandler
    {
        private int _requests;
        private int _listing;

        public int Requests => Volatile.Read(ref _requests);
        public int ListingRequests => Volatile.Read(ref _listing);

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Interlocked.Increment(ref _requests);
            var path = Uri.UnescapeDataString(request.RequestUri!.AbsolutePath);

            if (path == $"/api/models/{Repo}/tree/main")
            {
                Interlocked.Increment(ref _listing);
                var json = "[" + string.Join(",", Files.Select(f => $$"""{"path":"{{f.Key}}","type":"file","size":{{f.Value}}}""")) + "]";
                return Respond(HttpStatusCode.OK, System.Text.Encoding.UTF8.GetBytes(json));
            }

            var prefix = $"/{Repo}/resolve/main/";
            if (path.StartsWith(prefix, StringComparison.Ordinal) && Files.TryGetValue(path[prefix.Length..], out var size))
                return Respond(HttpStatusCode.OK, Enumerable.Repeat((byte)'x', size).ToArray());

            return Respond(HttpStatusCode.NotFound, []);
        }

        private static Task<HttpResponseMessage> Respond(HttpStatusCode status, byte[] body) =>
            Task.FromResult(new HttpResponseMessage(status) { Content = new ByteArrayContent(body) });
    }
}
