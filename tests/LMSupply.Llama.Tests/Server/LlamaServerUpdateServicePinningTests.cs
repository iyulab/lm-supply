using System.Net;
using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// <see cref="LlamaServerUpdateOptions.PinnedVersion"/>/<see cref="LlamaServerUpdateOptions.ServerBinaryPath"/>:
/// a pinned/externally-supplied installation must never call GitHub's "latest release" endpoint,
/// regardless of whether the pinned version is already cached. Network-free where possible (fake
/// HttpMessageHandler); the one cache-miss-download test builds a real platform-appropriate archive
/// so the extraction path is exercised too.
/// </summary>
public sealed class LlamaServerUpdateServicePinningTests : IDisposable
{
    private readonly string _dir = Path.Combine(
        Path.GetTempPath(), "lmsupply-pin-" + Guid.NewGuid().ToString("N"));

    public LlamaServerUpdateServicePinningTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort temp cleanup */ }
        GC.SuppressFinalize(this);
    }

    [Fact]
    public async Task PinnedVersion_CacheHit_MakesZeroNetworkCalls()
    {
        SeedCachedBinary(_dir, "b7898", LlamaServerBackend.Cpu);

        var handler = new NeverCallHandler();
        using var http = new HttpClient(handler);
        var options = new LlamaServerUpdateOptions { CacheDirectory = _dir, PinnedVersion = "b7898" };
        await using var service = new LlamaServerUpdateService(options, http);

        var result = await service.GetServerPathAsync(LlamaServerBackend.Cpu, cancellationToken: TestContext.Current.CancellationToken);

        result.Success.Should().BeTrue();
        result.NewVersion.Should().Be("b7898");
        File.Exists(result.ServerPath).Should().BeTrue();
        handler.CallCount.Should().Be(0, "a cache hit for a pinned version must never touch the network");
    }

    [Fact]
    public async Task PinnedVersion_CacheMiss_DownloadsExactTag_NeverCallsLatestEndpoint()
    {
        var assetName = FakeReleaseAssets.CpuAssetName("b7898");
        var downloadUrl = "https://fake.local/" + assetName;
        var releaseJson =
            $$"""{ "tag_name": "b7898", "assets": [ { "name": "{{assetName}}", "browser_download_url": "{{downloadUrl}}" } ] }""";
        var archiveBytes = FakeReleaseAssets.ServerArchive();

        var handler = new TagsOnlyHandler(downloadUrl, releaseJson, archiveBytes);
        using var http = new HttpClient(handler);
        var options = new LlamaServerUpdateOptions { CacheDirectory = _dir, PinnedVersion = "b7898" };
        await using var service = new LlamaServerUpdateService(options, http);

        var result = await service.GetServerPathAsync(LlamaServerBackend.Cpu, cancellationToken: TestContext.Current.CancellationToken);

        result.Success.Should().BeTrue();
        result.NewVersion.Should().Be("b7898");
        File.Exists(result.ServerPath).Should().BeTrue();
        handler.LatestEndpointHitCount.Should().Be(0,
            "a pinned version must resolve straight to its tag, never GET .../releases/latest");
    }

    [Fact]
    public async Task ServerBinaryPath_Set_SkipsAcquisitionEntirely_ZeroNetworkCalls()
    {
        var binaryPath = Path.Combine(_dir, "external-llama-server" + (OperatingSystem.IsWindows() ? ".exe" : ""));
        File.WriteAllBytes(binaryPath, "external binary"u8.ToArray());

        var handler = new NeverCallHandler();
        using var http = new HttpClient(handler);
        var options = new LlamaServerUpdateOptions { CacheDirectory = _dir, ServerBinaryPath = binaryPath };
        await using var service = new LlamaServerUpdateService(options, http);

        var result = await service.GetServerPathAsync(LlamaServerBackend.Cpu, cancellationToken: TestContext.Current.CancellationToken);

        result.Success.Should().BeTrue();
        result.ServerPath.Should().Be(binaryPath);
        handler.CallCount.Should().Be(0, "an externally-supplied binary must never touch the network");
    }

    [Fact]
    public async Task ServerBinaryPath_MissingFile_ReturnsFailedResult_DoesNotThrow()
    {
        var missingPath = Path.Combine(_dir, "does-not-exist.exe");
        var options = new LlamaServerUpdateOptions { CacheDirectory = _dir, ServerBinaryPath = missingPath };
        await using var service = new LlamaServerUpdateService(options);

        var result = await service.GetServerPathAsync(LlamaServerBackend.Cpu, cancellationToken: TestContext.Current.CancellationToken);

        result.Success.Should().BeFalse();
        result.Error.Should().Contain(missingPath);
    }

    [Fact]
    public async Task CheckAndApplyUpdateAsync_PinnedVersion_NeverCallsLatestEndpoint()
    {
        SeedCachedBinary(_dir, "b7898", LlamaServerBackend.Cpu);

        var handler = new NeverCallHandler();
        using var http = new HttpClient(handler);
        var options = new LlamaServerUpdateOptions { CacheDirectory = _dir, PinnedVersion = "b7898" };
        await using var service = new LlamaServerUpdateService(options, http);

        var result = await service.CheckAndApplyUpdateAsync(LlamaServerBackend.Cpu, cancellationToken: TestContext.Current.CancellationToken);

        result.Success.Should().BeTrue();
        result.Updated.Should().BeFalse("a pinned installation never auto-applies a newer version");
        handler.CallCount.Should().Be(0, "a pinned installation must never re-check for updates");
    }

    [Fact]
    public void Resolve_NullOptions_ReturnsProcessWideSingleton()
    {
        LlamaServerUpdateService.Resolve(null).Should().BeSameAs(LlamaServerUpdateService.Instance);
    }

    [Fact]
    public void Resolve_SameOptionsInstance_ReturnsSameService_DifferentInstance_ReturnsDifferentService()
    {
        var optionsA = new LlamaServerUpdateOptions { CacheDirectory = _dir };
        var optionsB = new LlamaServerUpdateOptions { CacheDirectory = _dir };

        var serviceA1 = LlamaServerUpdateService.Resolve(optionsA);
        var serviceA2 = LlamaServerUpdateService.Resolve(optionsA);
        var serviceB = LlamaServerUpdateService.Resolve(optionsB);

        serviceA2.Should().BeSameAs(serviceA1,
            "reusing the same options object must share one background-update timer and state file");
        serviceB.Should().NotBeSameAs(serviceA1, "distinct options objects must get isolated services");
    }

    private static void SeedCachedBinary(string cacheDir, string version, LlamaServerBackend backend)
    {
        var versionDir = Path.Combine(cacheDir, version, backend.ToString().ToLowerInvariant());
        Directory.CreateDirectory(versionDir);
        File.WriteAllBytes(Path.Combine(versionDir, FakeReleaseAssets.ServerExecutableName), "cached binary"u8.ToArray());
    }

    /// <summary>Fails the test outright if any request is made — proves a code path is network-free.</summary>
    private sealed class NeverCallHandler : HttpMessageHandler
    {
        public int CallCount { get; private set; }

        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken)
        {
            CallCount++;
            throw new InvalidOperationException(
                $"Unexpected network call to {request.RequestUri} — this path must be network-free.");
        }
    }

    /// <summary>Serves the pinned tag + download URL; fails if /releases/latest is ever hit.</summary>
    private sealed class TagsOnlyHandler(string downloadUrl, string releaseJson, byte[] archiveBytes)
        : HttpMessageHandler
    {
        public int LatestEndpointHitCount { get; private set; }

        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var url = request.RequestUri!.ToString();

            if (url.EndsWith("/releases/latest", StringComparison.Ordinal))
            {
                LatestEndpointHitCount++;
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.InternalServerError));
            }

            if (url.Contains("/releases/tags/", StringComparison.Ordinal))
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                { Content = new StringContent(releaseJson) });

            if (url == downloadUrl)
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                { Content = new ByteArrayContent(archiveBytes) });

            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
        }
    }
}
