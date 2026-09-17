using System.IO.Compression;
using System.Net;
using System.Runtime.InteropServices;
using AwesomeAssertions;
using LMSupply.Runtime;

namespace LMSupply.Core.Tests.Runtime;

/// <summary>
/// The native ONNX Runtime the downloader serves must be the version the feed says exists for the loaded
/// managed assembly (or the caller's), and a cached copy of some other version is only a stand-in for an
/// unreachable feed. Until 0.67.1 the downloader took any cached version before asking the feed, so a
/// machine that had once cached native 1.24.4 kept running managed 1.30.0 on it (dogfooding 2026-09-17).
/// </summary>
public sealed class OnnxNuGetDownloaderVersionPreferenceTests : IDisposable
{
    private const string PackageId = "microsoft.ml.onnxruntime";
    private static readonly PlatformInfo Windows = new()
    {
        OS = OSPlatform.Windows,
        Architecture = Architecture.X64,
        RuntimeIdentifier = "win-x64",
    };

    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-ort-version-pref-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    private string CachedNative(string version)
    {
        var dir = Path.Combine(_cacheDir, "onnxruntime", "cpu", version, "win-x64");
        Directory.CreateDirectory(dir);
        File.WriteAllBytes(Path.Combine(dir, "onnxruntime.dll"), [1, 2, 3]);
        return dir;
    }

    private static byte[] Nupkg()
    {
        using var stream = new MemoryStream();
        using (var zip = new ZipArchive(stream, ZipArchiveMode.Create, leaveOpen: true))
        {
            using var entry = zip.CreateEntry("runtimes/win-x64/native/onnxruntime.dll").Open();
            entry.Write([9, 9, 9]);
        }
        return stream.ToArray();
    }

    [Fact]
    public async Task RequestedVersionIsDownloaded_EvenWhenAnOlderVersionIsAlreadyCached()
    {
        CachedNative("1.24.4");
        var handler = new FeedHandler(versions: ["1.24.4", "1.30.0"], nupkg: Nupkg());
        using var downloader = new OnnxNuGetDownloader(_cacheDir, handler);

        var path = await downloader.DownloadAsync("cpu", Windows, version: "1.30.0", cancellationToken: TestContext.Current.CancellationToken);

        path.Should().Be(Path.Combine(_cacheDir, "onnxruntime", "cpu", "1.30.0", "win-x64"));
        File.Exists(Path.Combine(path, "onnxruntime.dll")).Should().BeTrue();
        handler.Requested.Should().Contain(u => u.EndsWith($"/{PackageId}/1.30.0/{PackageId}.1.30.0.nupkg"),
            "the requested version is downloaded; the cached 1.24.4 is not a substitute");
    }

    [Fact]
    public async Task FeedUnreachable_FallsBackToTheCachedVersion()
    {
        var cached = CachedNative("1.24.4");
        using var downloader = new OnnxNuGetDownloader(_cacheDir, new FeedHandler(offline: true));

        var path = await downloader.DownloadAsync("cpu", Windows, version: "1.30.0", cancellationToken: TestContext.Current.CancellationToken);

        path.Should().Be(cached, "with no feed, the cached copy is the only thing that can run");
    }

    [Fact]
    public async Task FeedUnreachable_PrefersTheNewestCachedVersion_ByVersionNotByString()
    {
        CachedNative("1.9.0");
        var newest = CachedNative("1.30.0");
        using var downloader = new OnnxNuGetDownloader(_cacheDir, new FeedHandler(offline: true));

        var path = await downloader.DownloadAsync("cpu", Windows, version: "1.31.0", cancellationToken: TestContext.Current.CancellationToken);

        path.Should().Be(newest, "\"1.9.0\" sorts after \"1.30.0\" as a string but before it as a version");
    }

    [Fact]
    public async Task FeedUnreachable_NothingCached_Throws()
    {
        using var downloader = new OnnxNuGetDownloader(_cacheDir, new FeedHandler(offline: true));

        var load = () => downloader.DownloadAsync("cpu", Windows, version: "1.30.0", cancellationToken: TestContext.Current.CancellationToken);

        await load.Should().ThrowAsync<InvalidOperationException>().WithMessage("*feed is unreachable*no cached version*");
    }

    /// <summary>A flat-container feed: an index of versions and one nupkg body, or an unreachable network.</summary>
    private sealed class FeedHandler(string[]? versions = null, byte[]? nupkg = null, bool offline = false) : HttpMessageHandler
    {
        public List<string> Requested { get; } = [];

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var url = request.RequestUri!.ToString();
            Requested.Add(url);
            if (offline)
                throw new HttpRequestException("No such host is known.");

            if (url.EndsWith("/index.json", StringComparison.Ordinal))
            {
                var json = "{\"versions\":[" + string.Join(",", (versions ?? []).Select(v => $"\"{v}\"")) + "]}";
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(json, System.Text.Encoding.UTF8, "application/json") });
            }

            if (url.EndsWith(".nupkg", StringComparison.Ordinal) && nupkg is not null)
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(nupkg) });

            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
        }
    }
}
