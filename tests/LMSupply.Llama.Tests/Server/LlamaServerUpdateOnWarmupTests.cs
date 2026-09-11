using System.Net;
using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// <see cref="LlamaServerUpdateOptions.UpdateOnWarmup"/> was documented to make a load wait for updates and
/// was read by nothing — every load started on the cached build whatever it was set to. It now decides
/// whether acquiring the server checks for a newer build first. Off by default, so a load never waits on
/// GitHub unless asked to. Network-free (fake <see cref="HttpMessageHandler"/>).
/// </summary>
public sealed class LlamaServerUpdateOnWarmupTests : IDisposable
{
    private const string CachedBuild = "b10200";
    private const string LatestBuild = "b10809";

    private readonly string _dir = Path.Combine(
        Path.GetTempPath(), "lmsupply-warmup-" + Guid.NewGuid().ToString("N"));

    public LlamaServerUpdateOnWarmupTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort temp cleanup */ }
        GC.SuppressFinalize(this);
    }

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    [Fact]
    public void IsOffByDefault()
    {
        new LlamaServerUpdateOptions().UpdateOnWarmup.Should().BeFalse();
    }

    [Fact]
    public async Task Off_TheLoadStartsOnTheCachedBuild_WithoutAskingGitHub()
    {
        await InstallCachedBuildAsync();

        var handler = new ReleasesHandler(reachable: true);
        using var http = new HttpClient(handler);
        await using var service = new LlamaServerUpdateService(Options(updateOnWarmup: false), http);

        var result = await service.GetServerPathAsync(LlamaServerBackend.Cpu, cancellationToken: Ct);

        result.Success.Should().BeTrue(result.Error);
        result.PreviousVersion.Should().Be(CachedBuild);
        handler.Requests.Should().BeEmpty("the load path must not wait on GitHub unless UpdateOnWarmup is set");
    }

    [Fact]
    public async Task On_TheLoadStartsOnTheNewerBuild()
    {
        await InstallCachedBuildAsync();

        var handler = new ReleasesHandler(reachable: true);
        using var http = new HttpClient(handler);
        await using var service = new LlamaServerUpdateService(Options(updateOnWarmup: true), http);

        var result = await service.GetServerPathAsync(LlamaServerBackend.Cpu, cancellationToken: Ct);

        result.Success.Should().BeTrue(result.Error);
        result.Updated.Should().BeTrue();
        result.NewVersion.Should().Be(LatestBuild);
        result.ServerPath.Should().StartWith(Path.Combine(_dir, LatestBuild));
        File.Exists(result.ServerPath).Should().BeTrue();
    }

    [Fact]
    public async Task On_WithGitHubUnreachable_TheLoadStartsOnTheCachedBuild()
    {
        await InstallCachedBuildAsync();

        var handler = new ReleasesHandler(reachable: false);
        using var http = new HttpClient(handler);
        await using var service = new LlamaServerUpdateService(Options(updateOnWarmup: true), http);

        var result = await service.GetServerPathAsync(LlamaServerBackend.Cpu, cancellationToken: Ct);

        result.Success.Should().BeTrue(result.Error);
        result.Updated.Should().BeFalse();
        result.ServerPath.Should().StartWith(Path.Combine(_dir, CachedBuild));
        handler.Requests.Should().NotBeEmpty("the check must have been attempted — or this test proves nothing");
    }

    [Fact]
    public async Task On_WithAPinnedVersion_NeverAsksGitHub()
    {
        SeedBinary(LatestBuild);

        var handler = new ReleasesHandler(reachable: true);
        using var http = new HttpClient(handler);
        var options = Options(updateOnWarmup: true);
        options.PinnedVersion = LatestBuild;
        await using var service = new LlamaServerUpdateService(options, http);

        var result = await service.GetServerPathAsync(LlamaServerBackend.Cpu, cancellationToken: Ct);

        result.Success.Should().BeTrue(result.Error);
        result.NewVersion.Should().Be(LatestBuild);
        handler.Requests.Should().BeEmpty("a pinned installation is never updated");
    }

    // Background downloads are off so the only requests a test sees are the load path's own.
    private LlamaServerUpdateOptions Options(bool updateOnWarmup) => new()
    {
        CacheDirectory = _dir,
        AutoDownloadUpdates = false,
        UpdateOnWarmup = updateOnWarmup,
    };

    // An older build, cached and recorded as the installed one — the state an earlier load leaves.
    private async Task InstallCachedBuildAsync()
    {
        SeedBinary(CachedBuild);
        using var http = new HttpClient(new ReleasesHandler(reachable: false));
        await using var service = new LlamaServerUpdateService(Options(updateOnWarmup: false), http);

        var installed = await service.GetServerPathAsync(LlamaServerBackend.Cpu, cancellationToken: Ct);

        installed.PreviousVersion.Should().Be(CachedBuild);
    }

    private void SeedBinary(string build)
    {
        var versionDir = Path.Combine(_dir, build, "cpu");
        Directory.CreateDirectory(versionDir);
        File.WriteAllBytes(Path.Combine(versionDir, FakeReleaseAssets.ServerExecutableName), "cached binary"u8.ToArray());
    }

    /// <summary>
    /// A GitHub whose latest release is the build release <see cref="LatestBuild"/>, with this platform's
    /// CPU asset — or, when unreachable, one that answers every request with 503. Records every request.
    /// </summary>
    private sealed class ReleasesHandler(bool reachable) : HttpMessageHandler
    {
        private const string FakeHost = "https://fake.local/";
        private readonly List<string> _requests = [];

        public IReadOnlyList<string> Requests
        {
            get
            {
                lock (_requests)
                    return [.. _requests];
            }
        }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var url = request.RequestUri!.ToString();
            lock (_requests)
                _requests.Add(url);

            if (!reachable)
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.ServiceUnavailable));

            var asset = FakeReleaseAssets.CpuAssetName(LatestBuild);
            if (url.EndsWith("/releases/latest", StringComparison.Ordinal)
                || url.EndsWith($"/releases/tags/{LatestBuild}", StringComparison.Ordinal))
            {
                var json = $$"""{ "tag_name": "{{LatestBuild}}", "prerelease": false, "assets": [ { "name": "{{asset}}", "browser_download_url": "{{FakeHost}}{{asset}}", "size": 42 } ] }""";
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                { Content = new StringContent(json, System.Text.Encoding.UTF8, "application/json") });
            }

            if (url == FakeHost + asset)
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                { Content = new ByteArrayContent(FakeReleaseAssets.ServerArchive()) });

            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
        }
    }
}
