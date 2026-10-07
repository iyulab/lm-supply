using System.IO.Compression;
using System.Net;
using System.Runtime.InteropServices;
using AwesomeAssertions;
using LMSupply.Runtime;

namespace LMSupply.Core.Tests.Runtime;

/// <summary>
/// Loads that start together on a cold cache (an image generator creating its sessions in parallel, an application
/// loading two models with Task.WhenAll) all ask for the same native runtime at once. They must converge on one
/// download, and a runtime already in the cache — possibly already loaded by another caller — must never be deleted
/// to make room for a second copy of itself.
/// </summary>
public sealed class OnnxNuGetDownloaderConcurrencyTests : IDisposable
{
    private static readonly PlatformInfo Windows = new()
    {
        OS = OSPlatform.Windows,
        Architecture = Architecture.X64,
        RuntimeIdentifier = "win-x64",
    };

    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-ort-concurrency-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    private string CachePath(string version) => Path.Combine(_cacheDir, "onnxruntime", "cpu", version, "win-x64");

    private static byte[] Nupkg(byte marker)
    {
        using var stream = new MemoryStream();
        using (var zip = new ZipArchive(stream, ZipArchiveMode.Create, leaveOpen: true))
        {
            using var entry = zip.CreateEntry("runtimes/win-x64/native/onnxruntime.dll").Open();
            entry.Write([marker, marker, marker]);
        }
        return stream.ToArray();
    }

    [Fact]
    public async Task ConcurrentFirstRunCalls_DownloadThePackageOnce_AndAllGetTheSameRuntime()
    {
        var handler = new SlowFeedHandler(Nupkg(7));
        using var downloader = new OnnxNuGetDownloader(_cacheDir, handler);
        var ct = TestContext.Current.CancellationToken;

        var paths = await Task.WhenAll(Enumerable.Range(0, 4).Select(_ =>
            Task.Run(() => downloader.DownloadAsync("cpu", Windows, version: "1.30.0", cancellationToken: ct), ct)));

        paths.Should().AllBe(CachePath("1.30.0"));
        handler.NupkgRequests.Should().Be(1, "callers that arrive while the runtime is being fetched wait for it instead of fetching it again");
        File.ReadAllBytes(Path.Combine(CachePath("1.30.0"), "onnxruntime.dll")).Should().Equal(7, 7, 7);
    }

    [Fact]
    public void Publishing_KeepsAValidRuntimeAnotherWriterAlreadyPlaced_AndDiscardsTheStagedCopy()
    {
        // Another process finished first: its runtime may already be loaded, so it stays exactly as it is.
        var target = CachePath("1.30.0");
        Directory.CreateDirectory(target);
        File.WriteAllBytes(Path.Combine(target, "onnxruntime.dll"), [1, 1, 1]);
        var staged = Path.Combine(_cacheDir, "staged");
        Directory.CreateDirectory(staged);
        File.WriteAllBytes(Path.Combine(staged, "onnxruntime.dll"), [2, 2, 2]);

        var published = OnnxNuGetDownloader.PublishStagedRuntime(staged, target, isValid: _ => true);

        published.Should().Be(target);
        File.ReadAllBytes(Path.Combine(target, "onnxruntime.dll")).Should().Equal(1, 1, 1);
        Directory.Exists(staged).Should().BeFalse("the losing copy is cleaned up");
    }

    [Fact]
    public void Publishing_ReplacesALeftoverThatIsNotAValidRuntime()
    {
        // An interrupted earlier attempt left a directory without the native library: it is not a runtime anyone loaded.
        var target = CachePath("1.30.0");
        Directory.CreateDirectory(target);
        File.WriteAllText(Path.Combine(target, "partial.tmp"), "x");
        var staged = Path.Combine(_cacheDir, "staged");
        Directory.CreateDirectory(staged);
        File.WriteAllBytes(Path.Combine(staged, "onnxruntime.dll"), [2, 2, 2]);

        var published = OnnxNuGetDownloader.PublishStagedRuntime(
            staged, target, isValid: dir => File.Exists(Path.Combine(dir, "onnxruntime.dll")));

        published.Should().Be(target);
        File.ReadAllBytes(Path.Combine(target, "onnxruntime.dll")).Should().Equal(2, 2, 2);
        File.Exists(Path.Combine(target, "partial.tmp")).Should().BeFalse();
    }

    [Fact]
    public async Task TheRuntimeIsStagedBesideTheCache_NotInTheSystemTempDirectory()
    {
        // A move from the system temp directory into a cache on another volume is not a rename and fails; staging
        // beside the cache keeps the final step an atomic rename wherever the cache lives.
        var handler = new SlowFeedHandler(Nupkg(7), onNupkg: () =>
        {
            Directory.EnumerateDirectories(Path.Combine(_cacheDir, "onnxruntime", "cpu", "1.30.0"), ".staging-*")
                .Should().ContainSingle("the package is downloaded into a staging directory next to its final place");
        });
        using var downloader = new OnnxNuGetDownloader(_cacheDir, handler);

        await downloader.DownloadAsync("cpu", Windows, version: "1.30.0", cancellationToken: TestContext.Current.CancellationToken);

        Directory.EnumerateDirectories(Path.Combine(_cacheDir, "onnxruntime", "cpu", "1.30.0"))
            .Should().Equal([CachePath("1.30.0")], "the staging directory is gone once the runtime is in place");
    }

    /// <summary>A feed that serves one nupkg after a delay long enough for concurrent callers to pile up.</summary>
    private sealed class SlowFeedHandler(byte[] nupkg, Action? onNupkg = null) : HttpMessageHandler
    {
        private int _nupkgRequests;

        public int NupkgRequests => Volatile.Read(ref _nupkgRequests);

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var url = request.RequestUri!.ToString();
            if (!url.EndsWith(".nupkg", StringComparison.Ordinal))
                return new HttpResponseMessage(HttpStatusCode.NotFound);

            Interlocked.Increment(ref _nupkgRequests);
            onNupkg?.Invoke();
            await Task.Delay(200, cancellationToken);
            return new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(nupkg) };
        }
    }
}

