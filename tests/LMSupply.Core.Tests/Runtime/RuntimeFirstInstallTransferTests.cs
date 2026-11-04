using System.Runtime.InteropServices;
using AwesomeAssertions;
using LMSupply.Download;
using LMSupply.Runtime;

namespace LMSupply.Core.Tests.Runtime;

/// <summary>
/// A first load transfers what its caller agreed to — the runtime and the model — and nothing else: the load that
/// installs a runtime does not start a background update download of its own, and what an interrupted background download
/// left in the cache is swept instead of staying for good.
/// </summary>
[Trait("Category", "Unit")]
public sealed class RuntimeFirstInstallTransferTests : IDisposable
{
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), $"lmsupply-first-install-{Guid.NewGuid():N}");

    private static readonly PlatformInfo Platform = new()
    {
        OS = OSPlatform.Windows, Architecture = Architecture.X64, RuntimeIdentifier = "win-x64"
    };

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_cacheDir))
                Directory.Delete(_cacheDir, recursive: true);
        }
        catch (IOException)
        {
            // Best effort.
        }
    }

    private sealed class CountingFeed : HttpMessageHandler
    {
        private int _requests;
        public int Requests => Volatile.Read(ref _requests);

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Interlocked.Increment(ref _requests);
            return Task.FromResult(new HttpResponseMessage(System.Net.HttpStatusCode.NotFound));
        }
    }

    private static Task<string> Install(string version, IProgress<DownloadProgress>? progress, CancellationToken ct) =>
        Task.FromResult("/runtime/" + version);

    [Fact]
    public async Task TheLoadThatInstallsARuntime_AsksNugetForNothing()
    {
        var feed = new CountingFeed();
        using var http = new HttpClient(feed);
        await using var service = new RuntimeUpdateService(
            $"first-{Guid.NewGuid():N}", new RuntimeUpdateOptions { CacheDirectory = _cacheDir }, new NuGetPackageResolver(http));

        await service.GetRuntimePathAsync("Some.Runtime", "cpu", Platform, "1.0.0", Install, ct: Ct);
        await Task.Delay(600, Ct);   // a background check, if one were started, issues its request within milliseconds

        feed.Requests.Should().Be(0, "a runtime installed a moment ago is as new as this host asks for; the first check is one interval away");
    }

    [Fact]
    public async Task ALaterLoad_OnceTheIntervalHasPassed_StillChecksInTheBackground()
    {
        // Positive control for the test above: the background check exists, and a due one does ask.
        var feed = new CountingFeed();
        using var http = new HttpClient(feed);
        var type = $"later-{Guid.NewGuid():N}";
        await using var service = new RuntimeUpdateService(
            type, new RuntimeUpdateOptions { CacheDirectory = _cacheDir }, new NuGetPackageResolver(http));
        await service.GetRuntimePathAsync("Some.Runtime", "cpu", Platform, "1.0.0", Install, ct: Ct);

        using var states = new RuntimeVersionStateManager(_cacheDir);
        var key = RuntimeVersionStateManager.GetPackageKey(type, "cpu", Platform.RuntimeIdentifier);
        var state = (await states.GetStateAsync(key, Ct))!;
        state.LastVersionCheck = DateTimeOffset.UtcNow - TimeSpan.FromDays(2);
        await states.UpdateStateAsync(key, state, Ct);

        // A later process: a new service over the same cache directory reads the state from disk.
        await using var later = new RuntimeUpdateService(
            type, new RuntimeUpdateOptions { CacheDirectory = _cacheDir }, new NuGetPackageResolver(http));
        await later.GetRuntimePathAsync("Some.Runtime", "cpu", Platform, "1.0.0", Install, ct: Ct);
        var deadline = DateTime.UtcNow.AddSeconds(10);
        while (feed.Requests == 0 && DateTime.UtcNow < deadline)
            await Task.Delay(50, Ct);

        feed.Requests.Should().BeGreaterThan(0);
    }

    [Fact]
    public void AStagingDirectoryNothingHasWrittenToForADay_IsSwept_ALiveOneIsNot()
    {
        Directory.CreateDirectory(_cacheDir);
        var now = DateTime.UtcNow;
        string Staging(string name, DateTime fileWritten, DateTime? dirWritten = null)
        {
            var dir = Path.Combine(_cacheDir, name);
            Directory.CreateDirectory(dir);
            var file = Path.Combine(dir, "Pkg.1.0.0.nupkg.part");
            File.WriteAllBytes(file, new byte[10]);
            File.SetLastWriteTimeUtc(file, fileWritten);
            Directory.SetLastWriteTimeUtc(dir, dirWritten ?? fileWritten);
            return dir;
        }

        var abandoned = Staging(".staging-abandoned", now.AddDays(-2));
        var fresh = Staging(".staging-fresh", now.AddMinutes(-5));
        // Created long ago, but its download is still writing the file.
        var live = Staging(".staging-live", now.AddSeconds(-30), dirWritten: now.AddDays(-2));
        var other = Path.Combine(_cacheDir, "cpu");
        Directory.CreateDirectory(other);
        Directory.SetLastWriteTimeUtc(other, now.AddDays(-30));

        OnnxNuGetDownloader.SweepAbandonedStaging(_cacheDir, now);

        Directory.Exists(abandoned).Should().BeFalse();
        Directory.Exists(fresh).Should().BeTrue();
        Directory.Exists(live).Should().BeTrue("a download that is still writing is not abandoned however old its directory is");
        Directory.Exists(other).Should().BeTrue("only .staging-* directories are swept");
    }
}
