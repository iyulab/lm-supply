using System.Net;
using AwesomeAssertions;
using LMSupply.Runtime;

namespace LMSupply.Core.Tests.Runtime;

/// <summary>
/// What a load's runtime still needs, asked before the load: <see cref="RuntimeManager.IsRuntimeAvailableOfflineAsync"/> reads
/// the cache as the load would take from it, and <see cref="RuntimeManager.GetRuntimeDownloadBytesAsync"/> asks the feed for the
/// package's length without downloading it — null, not 0, when it cannot tell.
/// </summary>
[Trait("Category", "Unit")]
public sealed class RuntimeManagerRemainingDownloadTests : IDisposable
{
    private const string OnnxRuntime = RuntimePackageRegistry.PackageTypes.OnnxRuntime;
    private readonly string _root = Path.Combine(Path.GetTempPath(), "lmsupply-ort-remaining-" + Guid.NewGuid().ToString("N"));

    private string CacheDir => Path.Combine(_root, "cache");

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_root))
            Directory.Delete(_root, recursive: true);
    }

    private static async Task<RuntimeManager> CreateAsync(RuntimeManagerOptions options, Feed feed)
    {
        var manager = new RuntimeManager(options, updateOptions: null, feed);
        await manager.InitializeAsync(Ct);
        return manager;
    }

    private void SeedCache(RuntimeManager manager, string version)
    {
        var dir = Path.Combine(CacheDir, OnnxRuntime, "cpu", version, manager.Platform.RuntimeIdentifier);
        Directory.CreateDirectory(dir);
        File.WriteAllBytes(Path.Combine(dir, RuntimePackageRegistry.GetNativeLibraryFileName("onnxruntime", manager.Platform)), [1]);
    }

    [Fact]
    public async Task A_cached_version_a_load_asks_for_leaves_nothing_to_download_and_asks_no_feed()
    {
        var feed = new Feed(length: 1);
        var manager = await CreateAsync(new RuntimeManagerOptions { CacheDirectory = CacheDir, PinnedVersion = "1.30.0" }, feed);
        SeedCache(manager, "1.30.0");

        (await manager.IsRuntimeAvailableOfflineAsync("onnxruntime", "cpu", cancellationToken: Ct)).Should().BeTrue();
        (await manager.GetRuntimeDownloadBytesAsync("onnxruntime", "cpu", cancellationToken: Ct)).Should().Be(0);
        feed.Requests.Should().BeEmpty();
    }

    [Fact]
    public async Task Another_cached_version_does_not_stand_in_for_a_load_that_may_download()
    {
        var feed = new Feed(length: 61_000_000);
        var manager = await CreateAsync(new RuntimeManagerOptions { CacheDirectory = CacheDir, PinnedVersion = "1.30.0" }, feed);
        SeedCache(manager, "1.24.4");

        (await manager.IsRuntimeAvailableOfflineAsync("onnxruntime", "cpu", cancellationToken: Ct)).Should().BeFalse(
            "the load would download the version it asks for rather than run the cached one");
        (await manager.GetRuntimeDownloadBytesAsync("onnxruntime", "cpu", cancellationToken: Ct)).Should().Be(61_000_000);
        feed.Requests.Should().ContainSingle().Which.Should().Be("HEAD microsoft.ml.onnxruntime.1.30.0.nupkg", "the length is read, not the package");
        Directory.EnumerateFiles(CacheDir, "*.nupkg", SearchOption.AllDirectories).Should().BeEmpty();
    }

    [Fact]
    public async Task With_downloads_disabled_any_cached_version_stands_in_as_it_does_for_the_load()
    {
        var manager = await CreateAsync(new RuntimeManagerOptions { CacheDirectory = CacheDir, DisableAutoDownload = true }, new Feed(length: 1));
        SeedCache(manager, "1.24.4");

        (await manager.IsRuntimeAvailableOfflineAsync("onnxruntime", "cpu", "1.30.0", Ct)).Should().BeTrue();
    }

    [Fact]
    public async Task A_runtime_a_load_would_not_download_is_unknown_rather_than_zero()
    {
        var offline = await CreateAsync(new RuntimeManagerOptions { CacheDirectory = CacheDir, DisableAutoDownload = true }, new Feed(length: 1));
        (await offline.GetRuntimeDownloadBytesAsync("onnxruntime", "cpu", "1.30.0", Ct)).Should().BeNull("such a load fails instead");

        var bundled = await CreateAsync(new RuntimeManagerOptions { CacheDirectory = CacheDir, RuntimeDirectory = Path.Combine(_root, "empty") }, new Feed(length: 1));
        (await bundled.IsRuntimeAvailableOfflineAsync("onnxruntime", "cpu", cancellationToken: Ct)).Should().BeFalse();
        (await bundled.GetRuntimeDownloadBytesAsync("onnxruntime", "cpu", cancellationToken: Ct)).Should().BeNull();
    }

    [Theory]
    [InlineData(HttpStatusCode.ServiceUnavailable)]
    [InlineData(HttpStatusCode.OK)] // without a stated length
    public async Task A_feed_that_cannot_tell_the_length_is_unknown(HttpStatusCode status)
    {
        var manager = await CreateAsync(new RuntimeManagerOptions { CacheDirectory = CacheDir, PinnedVersion = "1.30.0" }, new Feed(length: null, status));

        (await manager.GetRuntimeDownloadBytesAsync("onnxruntime", "cpu", cancellationToken: Ct)).Should().BeNull();
    }

    /// <summary>Answers a package request with a length (or none), and records each as «METHOD file».</summary>
    private sealed class Feed(long? length, HttpStatusCode status = HttpStatusCode.OK) : HttpMessageHandler
    {
        public List<string> Requests { get; } = [];

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            lock (Requests)
                Requests.Add($"{request.Method} {request.RequestUri!.Segments[^1]}");
            var response = new HttpResponseMessage(status) { Content = new ByteArrayContent([]) };
            response.Content.Headers.ContentLength = length;
            return Task.FromResult(response);
        }
    }
}
