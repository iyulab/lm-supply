using System.Net;
using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// A llama-server build below <see cref="LlamaServerVersionRequirements.MinimumSecureBuild"/> is replaced by the latest
/// build when nothing is pinned and downloads are allowed, refused otherwise — pinned, supplied, cached with downloads
/// disabled, or rolled back to — and run only when <see cref="LlamaServerUpdateOptions.AllowVulnerableBuild"/> is set.
/// </summary>
public sealed class LlamaServerSecurityFloorTests : IDisposable
{
    private const string Old = "b8000";
    private const string Latest = "b9100";
    private const string FakeHost = "https://fake.local/";

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "lmsupply-floor-" + Guid.NewGuid().ToString("N"));

    public LlamaServerSecurityFloorTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort temp cleanup */ }
    }

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    [Theory]
    [InlineData("b8145", true)]
    [InlineData("b8146", false)]
    [InlineData("b11476", false)]
    [InlineData("external", false)]
    [InlineData(null, false)]
    public void TheFloor_IsTheNewestFixAmongThePublishedAdvisories(string? version, bool below)
    {
        LlamaServerVersionRequirements.MinimumSecureBuild.Should().Be(8146);
        LlamaServerVersionRequirements.IsBelowSecurityFloor(version).Should().Be(below);
    }

    [Fact]
    public async Task AnOldCachedBuild_IsReplacedByTheLatest_WhenDownloadsAreAllowed()
    {
        SeedCachedBinary(Old);
        using var http = new HttpClient(new Hub());
        await using var service = new LlamaServerUpdateService(new LlamaServerUpdateOptions { CacheDirectory = _dir, AutoDownloadUpdates = false }, http);

        var result = await service.GetServerPathAsync(LlamaServerBackend.Cpu, cancellationToken: Ct);

        result.Success.Should().BeTrue(result.Error);
        (result.NewVersion ?? result.PreviousVersion).Should().Be(Latest);
    }

    [Fact]
    public async Task AnOldCachedBuild_ReadsAsNoServerCached_WhenDownloadsAreDisabled()
    {
        SeedCachedBinary(Old);
        var hub = new Hub();
        using var http = new HttpClient(hub);
        await using var service = new LlamaServerUpdateService(new LlamaServerUpdateOptions { CacheDirectory = _dir }, http);

        var act = () => service.AcquireServerAsync(LlamaServerBackend.Cpu, localFilesOnly: true, cancellationToken: Ct);

        // The answer an empty cache gets, so a check of what is available offline says no and a download brings the latest.
        (await act.Should().ThrowAsync<LMSupply.Exceptions.ModelNotFoundException>()).Which.Message
            .Should().Contain(Old).And.Contain("b8146").And.Contain("AllowVulnerableBuild");
        hub.Requests.Should().Be(0);
    }

    [Fact]
    public async Task AnOldCachedBuild_IsRefused_WhenTheLatestCannotBeAcquired()
    {
        SeedCachedBinary(Old);
        using var http = new HttpClient(new Hub(unreachable: true));
        await using var service = new LlamaServerUpdateService(new LlamaServerUpdateOptions { CacheDirectory = _dir, AutoDownloadUpdates = false }, http);

        var result = await service.GetServerPathAsync(LlamaServerBackend.Cpu, cancellationToken: Ct);

        result.Success.Should().BeFalse("the old build is not run in its place");
        result.Error.Should().Contain("could not be acquired");
    }

    [Fact]
    public async Task APinnedOldBuild_IsRefused_WithoutARequest()
    {
        SeedCachedBinary(Old);
        var hub = new Hub();
        using var http = new HttpClient(hub);
        await using var service = new LlamaServerUpdateService(new LlamaServerUpdateOptions { CacheDirectory = _dir, PinnedVersion = Old }, http);

        var result = await service.GetServerPathAsync(LlamaServerBackend.Cpu, cancellationToken: Ct);

        result.Success.Should().BeFalse();
        result.Error.Should().Contain("Pin or supply a newer build");
        hub.Requests.Should().Be(0, "a pin is never exceeded, so nothing is looked up");
    }

    [Fact]
    public async Task ASuppliedBinaryDeclaredOld_IsRefused()
    {
        var binary = Path.Combine(_dir, FakeReleaseAssets.ServerExecutableName);
        await File.WriteAllBytesAsync(binary, "supplied binary"u8.ToArray(), Ct);
        await using var service = new LlamaServerUpdateService(new LlamaServerUpdateOptions { CacheDirectory = _dir, ServerBinaryPath = binary, PinnedVersion = Old });

        (await service.GetServerPathAsync(LlamaServerBackend.Cpu, cancellationToken: Ct)).Success.Should().BeFalse();
    }

    [Fact]
    public async Task AnOldBuild_IsRun_WhenTheOptionsAllowIt()
    {
        SeedCachedBinary(Old);
        using var http = new HttpClient(new Hub());
        await using var service = new LlamaServerUpdateService(
            new LlamaServerUpdateOptions { CacheDirectory = _dir, PinnedVersion = Old, AllowVulnerableBuild = true }, http);

        var installation = await service.AcquireServerAsync(LlamaServerBackend.Cpu, localFilesOnly: false, cancellationToken: Ct);

        installation.Version.Should().Be(Old);
    }

    [Fact]
    public async Task ARollback_PassesOverABuildBelowTheFloor_UnlessTheOptionsAllowIt()
    {
        var platform = LlamaServerUpdateService.GetCurrentPlatform();
        using (var manager = new LlamaServerStateManager(_dir))
        {
            await manager.CreateInitialStateAsync("cpu", platform, Old, "cpu", Ct);
            await manager.MarkUpdateReadyAsync("cpu", platform, "b8200", "cpu", Ct);
            await manager.ActivateUpdateAsync("cpu", platform, cancellationToken: Ct);
            await manager.MarkUpdateReadyAsync("cpu", platform, Latest, "cpu", Ct);
            await manager.ActivateUpdateAsync("cpu", platform, cancellationToken: Ct);
            foreach (var version in (string[])[Old, "b8200", Latest])
            {
                var directory = manager.GetBuildDirectory(version, "cpu");
                Directory.CreateDirectory(directory);
                await File.WriteAllBytesAsync(Path.Combine(directory, FakeReleaseAssets.ServerExecutableName), "build"u8.ToArray(), Ct);
            }
        }

        await using (var service = new LlamaServerUpdateService(new LlamaServerUpdateOptions { CacheDirectory = _dir, AutoDownloadUpdates = false }))
        {
            (await service.RollbackAsync(LlamaServerBackend.Cpu, Ct)).PreviousVersion.Should().Be("b8200");

            var refused = await service.RollbackAsync(LlamaServerBackend.Cpu, Ct);
            refused.Success.Should().BeFalse();
            refused.Error.Should().Contain("b8146");
        }

        await using (var allowing = new LlamaServerUpdateService(new LlamaServerUpdateOptions { CacheDirectory = _dir, AutoDownloadUpdates = false, AllowVulnerableBuild = true }))
        {
            (await allowing.RollbackAsync(LlamaServerBackend.Cpu, Ct)).PreviousVersion.Should().Be(Old);
        }
    }

    private void SeedCachedBinary(string version)
    {
        var versionDir = Path.Combine(_dir, version, "cpu");
        Directory.CreateDirectory(versionDir);
        File.WriteAllBytes(Path.Combine(versionDir, FakeReleaseAssets.ServerExecutableName), "cached binary"u8.ToArray());
    }

    /// <summary>A one-build GitHub whose latest release is <see cref="Latest"/>, counting the requests it answers — or failing each one.</summary>
    private sealed class Hub(bool unreachable = false) : HttpMessageHandler
    {
        private int _requests;

        public int Requests => _requests;

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Interlocked.Increment(ref _requests);
            if (unreachable)
                throw new HttpRequestException("unreachable");
            var url = request.RequestUri!.ToString();
            var asset = FakeReleaseAssets.CpuAssetName(Latest);
            if (url == FakeHost + asset)
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(FakeReleaseAssets.ServerArchive()) });

            if (url.Contains("/releases", StringComparison.Ordinal))
            {
                var json = $$"""{ "tag_name": "{{Latest}}", "prerelease": false, "assets": [{ "name": "{{asset}}", "browser_download_url": "{{FakeHost}}{{asset}}", "size": {{FakeReleaseAssets.ServerArchiveSize}} }] }""";
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(json, System.Text.Encoding.UTF8, "application/json") });
            }

            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
        }
    }
}
