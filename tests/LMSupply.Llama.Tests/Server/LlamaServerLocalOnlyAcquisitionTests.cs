using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// A local-only acquisition (what a load with <c>DisableAutoDownload</c> asks for) makes no network request at all: a cached
/// server is used without the background update check, and a server that is not cached gives a
/// <see cref="LMSupply.Exceptions.ModelNotFoundException"/> instead of a download.
/// </summary>
public sealed class LlamaServerLocalOnlyAcquisitionTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "lmsupply-localonly-" + Guid.NewGuid().ToString("N"));

    public LlamaServerLocalOnlyAcquisitionTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort temp cleanup */ }
        GC.SuppressFinalize(this);
    }

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    // The background check waits a second before it asks the release feed; this is comfortably past it.
    private static readonly TimeSpan PastBackgroundCheck = TimeSpan.FromMilliseconds(2500);

    [Fact]
    public async Task NothingCached_ThrowsModelNotFound_WithoutARequest()
    {
        var handler = new CountingHandler();
        using var http = new HttpClient(handler);
        await using var service = new LlamaServerUpdateService(new LlamaServerUpdateOptions { CacheDirectory = _dir }, http);

        var act = () => service.AcquireServerAsync(LlamaServerBackend.Cpu, localFilesOnly: true, cancellationToken: Ct);

        (await act.Should().ThrowAsync<LMSupply.Exceptions.ModelNotFoundException>())
            .Which.Should().Match<LMSupply.Exceptions.ModelNotFoundException>(e => e.ModelId == "llama-server" && e.Message.Contains("DisableAutoDownload"));
        handler.Count.Should().Be(0);
    }

    [Fact]
    public async Task PinnedButNotCached_ThrowsModelNotFound_WithoutARequest()
    {
        var handler = new CountingHandler();
        using var http = new HttpClient(handler);
        await using var service = new LlamaServerUpdateService(
            new LlamaServerUpdateOptions { CacheDirectory = _dir, PinnedVersion = "b7898" }, http);

        var act = () => service.AcquireServerAsync(LlamaServerBackend.Cpu, localFilesOnly: true, cancellationToken: Ct);

        (await act.Should().ThrowAsync<LMSupply.Exceptions.ModelNotFoundException>()).Which.Message.Should().Contain("b7898");
        handler.Count.Should().Be(0);
    }

    [Fact]
    public async Task ACachedServer_IsUsed_AndNoBackgroundCheckFollows()
    {
        SeedCachedBinary(_dir, "b7898", LlamaServerBackend.Cpu);
        var handler = new CountingHandler();
        using var http = new HttpClient(handler);
        await using var service = new LlamaServerUpdateService(
            new LlamaServerUpdateOptions { CacheDirectory = _dir, AutoDownloadUpdates = true, UpdateOnWarmup = true, VersionCheckInterval = TimeSpan.Zero }, http);

        var installation = await service.AcquireServerAsync(LlamaServerBackend.Cpu, localFilesOnly: true, cancellationToken: Ct);
        await Task.Delay(PastBackgroundCheck, Ct);

        File.Exists(installation.ServerPath).Should().BeTrue();
        installation.Version.Should().Be("b7898");
        handler.Count.Should().Be(0, "neither the warm-up update nor the background check may ask the release feed");
    }

    [Fact]
    public async Task PositiveControl_WithoutLocalOnly_TheSameCachedServerTriggersTheBackgroundCheck()
    {
        SeedCachedBinary(_dir, "b7898", LlamaServerBackend.Cpu);
        var handler = new CountingHandler();
        using var http = new HttpClient(handler);
        await using var service = new LlamaServerUpdateService(
            new LlamaServerUpdateOptions { CacheDirectory = _dir, AutoDownloadUpdates = true, VersionCheckInterval = TimeSpan.Zero }, http);

        var result = await service.GetServerPathAsync(LlamaServerBackend.Cpu, cancellationToken: Ct);
        await Task.Delay(PastBackgroundCheck, Ct);

        result.Success.Should().BeTrue();
        handler.Count.Should().BeGreaterThan(0, "this is the request the local-only path must not make");
    }

    private static void SeedCachedBinary(string cacheDir, string version, LlamaServerBackend backend)
    {
        var versionDir = Path.Combine(cacheDir, version, backend.ToString().ToLowerInvariant());
        Directory.CreateDirectory(versionDir);
        File.WriteAllBytes(Path.Combine(versionDir, FakeReleaseAssets.ServerExecutableName), "cached binary"u8.ToArray());
    }

    /// <summary>Counts requests and fails each one, so a path that asks the network is visible without reaching it.</summary>
    private sealed class CountingHandler : HttpMessageHandler
    {
        private int _count;
        public int Count => Volatile.Read(ref _count);

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Interlocked.Increment(ref _count);
            return Task.FromResult(new HttpResponseMessage(System.Net.HttpStatusCode.ServiceUnavailable));
        }
    }
}
