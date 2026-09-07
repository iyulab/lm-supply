using System.Net;
using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// "Latest" resolution against llama.cpp's release scheme since 2026-08-21: <c>releases/latest</c>
/// is a versioned release (<c>vX.Y.Z</c>) whose only asset is <c>nightly-tag.txt</c> naming the
/// <c>bNNNNN</c> build it corresponds to, while the build releases themselves are marked prerelease.
/// Acquisition must always end on a build tag — that is what the asset names, the cache layout and
/// <see cref="LlamaServerVersionRequirements.ParseBuildNumber"/> are keyed on. Fully network-free
/// (fake <see cref="HttpMessageHandler"/>).
/// </summary>
public sealed class LlamaServerLatestResolutionTests : IDisposable
{
    private const string StableBuild = "b10809";   // what v0.4.0's nightly-tag.txt points at
    private const string NewestNightly = "b10840"; // newest prerelease build release
    private const string OlderNightly = "b10839";

    private readonly string _dir = Path.Combine(
        Path.GetTempPath(), "lmsupply-latest-" + Guid.NewGuid().ToString("N"));

    public LlamaServerLatestResolutionTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort temp cleanup */ }
        GC.SuppressFinalize(this);
    }

    [Fact]
    public async Task Latest_IsVersionedRelease_ResolvesBuildViaNightlyTag_AndDownloadsThatBuild()
    {
        var handler = new FakeReleasesHandler(latestTag: "v0.4.0", latestAssets: [FakeReleasesHandler.NightlyTagAsset]);
        using var http = new HttpClient(handler);
        var options = new LlamaServerUpdateOptions { CacheDirectory = _dir };
        await using var service = new LlamaServerUpdateService(options, http);

        var result = await service.GetServerPathAsync(LlamaServerBackend.Cpu, cancellationToken: TestContext.Current.CancellationToken);

        result.Success.Should().BeTrue(result.Error);
        result.NewVersion.Should().Be(StableBuild, "the versioned release must be resolved to the build it names");
        result.ServerPath.Should().StartWith(Path.Combine(_dir, StableBuild),
            "the cache directory is keyed by build tag, never by the versioned release tag");
        File.Exists(result.ServerPath).Should().BeTrue();

        var state = await File.ReadAllTextAsync(Path.Combine(_dir, "llama-server-state.json"), TestContext.Current.CancellationToken);
        state.Should().Contain(StableBuild).And.NotContain("v0.4.0",
            "the state file must record the build tag or ParseBuildNumber breaks on the next load");

        handler.Requests.Should().NotContain(u => u.EndsWith("/releases/tags/v0.4.0", StringComparison.Ordinal),
            "the latest-release payload already carries the nightly-tag asset; a second tag lookup is waste");
        handler.Requests.Should().NotContain(u => u.Contains("/releases?", StringComparison.Ordinal),
            "the listing fallback must not run when nightly-tag.txt resolved the build");
    }

    [Fact]
    public async Task Latest_IsVersionedRelease_WithoutNightlyTag_FallsBackToNewestBuildRelease()
    {
        var handler = new FakeReleasesHandler(latestTag: "v0.5.0", latestAssets: []);
        using var http = new HttpClient(handler);
        var options = new LlamaServerUpdateOptions { CacheDirectory = _dir };
        await using var service = new LlamaServerUpdateService(options, http);

        var result = await service.GetServerPathAsync(LlamaServerBackend.Cpu, cancellationToken: TestContext.Current.CancellationToken);

        result.Success.Should().BeTrue(result.Error);
        result.NewVersion.Should().Be(NewestNightly,
            "with no pointer asset the newest build release that actually has assets is the only sane answer, prerelease or not");
    }

    [Fact]
    public async Task Latest_IsBuildRelease_UsedDirectly_WithoutListing()
    {
        // Pre-2026-08 scheme (and any future return to it): latest itself is a build release.
        var handler = new FakeReleasesHandler(latestTag: StableBuild, latestAssets: [FakeReleasesHandler.CpuAssetFor(StableBuild)]);
        using var http = new HttpClient(handler);
        var options = new LlamaServerUpdateOptions { CacheDirectory = _dir };
        await using var service = new LlamaServerUpdateService(options, http);

        var result = await service.GetServerPathAsync(LlamaServerBackend.Cpu, cancellationToken: TestContext.Current.CancellationToken);

        result.Success.Should().BeTrue(result.Error);
        result.NewVersion.Should().Be(StableBuild);
        handler.Requests.Should().NotContain(u => u.Contains("/releases?", StringComparison.Ordinal));
        handler.Requests.Should().NotContain(u => u.EndsWith("nightly-tag.txt", StringComparison.Ordinal));
    }

    [Fact]
    public async Task IncludePrerelease_FollowsNewestBuildRelease_NeverAsksForLatest()
    {
        var handler = new FakeReleasesHandler(latestTag: "v0.4.0", latestAssets: [FakeReleasesHandler.NightlyTagAsset]);
        using var http = new HttpClient(handler);
        var options = new LlamaServerUpdateOptions { CacheDirectory = _dir, IncludePrerelease = true };
        await using var service = new LlamaServerUpdateService(options, http);

        var result = await service.GetServerPathAsync(LlamaServerBackend.Cpu, cancellationToken: TestContext.Current.CancellationToken);

        result.Success.Should().BeTrue(result.Error);
        result.NewVersion.Should().Be(NewestNightly, "IncludePrerelease opts into the nightly build line");
        handler.Requests.Should().NotContain(u => u.EndsWith("/releases/latest", StringComparison.Ordinal),
            "the stable pointer is irrelevant when the caller asked for prereleases");
    }

    [Fact]
    public async Task GetLatestVersionAsync_AlwaysReturnsBuildTag()
    {
        var handler = new FakeReleasesHandler(latestTag: "v0.4.0", latestAssets: [FakeReleasesHandler.NightlyTagAsset]);
        using var http = new HttpClient(handler);
        using var downloader = new LlamaServerDownloader(_dir, http);

        var latest = await downloader.GetLatestVersionAsync(TestContext.Current.CancellationToken);

        latest.Should().Be(StableBuild);
        LlamaServerVersionRequirements.ParseBuildNumber(latest).Should().Be(10809);
    }

    [Fact]
    public async Task GetAssetAsync_ExplicitVersionedTag_IsNormalizedToItsBuild()
    {
        // A caller (e.g. PinnedVersion) naming the versioned release must land on the same build
        // the auto path would — one normalization point, no "v0.4.0" cache directories.
        var handler = new FakeReleasesHandler(latestTag: "v0.4.0", latestAssets: [FakeReleasesHandler.NightlyTagAsset]);
        using var http = new HttpClient(handler);
        using var downloader = new LlamaServerDownloader(_dir, http);

        var asset = await downloader.GetAssetAsync("v0.4.0", LlamaServerBackend.Cpu, TestContext.Current.CancellationToken);

        asset.Should().NotBeNull();
        asset!.Version.Should().Be(StableBuild);
        asset.Name.Should().Be(FakeReleaseAssets.CpuAssetName(StableBuild));
    }

    [Fact]
    public void GetCachedVersions_OrdersByBuildNumber_NotLexically()
    {
        // "b9999" sorts after "b10809" as a string; the newest cached build must still win.
        foreach (var build in new[] { "b9999", "b10809", "b10200" })
            Directory.CreateDirectory(Path.Combine(_dir, build, "cpu"));
        using var downloader = new LlamaServerDownloader(_dir, new HttpClient(new FakeReleasesHandler("b1", [])));

        var versions = downloader.GetCachedVersions();

        versions.Should().Equal("b10809", "b10200", "b9999");
    }

    /// <summary>
    /// Fake GitHub Releases API. Serves: <c>/releases/latest</c> (configurable tag + assets),
    /// the <c>nightly-tag.txt</c> download (→ <see cref="StableBuild"/>), <c>/releases/tags/{build}</c>
    /// for the three known builds (each with this platform's CPU asset), the <c>/releases?…</c>
    /// listing (a versioned release without assets, then the two nightly prerelease builds, newest
    /// first), and the asset downloads. Records every request URL.
    /// </summary>
    private sealed class FakeReleasesHandler(string latestTag, string[] latestAssets) : HttpMessageHandler
    {
        private const string FakeHost = "https://fake.local/";
        public const string NightlyTagAsset = "nightly-tag.txt";

        public List<string> Requests { get; } = [];

        public static string CpuAssetFor(string build) => FakeReleaseAssets.CpuAssetName(build);

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var url = request.RequestUri!.ToString();
            Requests.Add(url);

            if (url.EndsWith("/releases/latest", StringComparison.Ordinal))
                return Json(Release(latestTag, prerelease: false, latestAssets));

            if (url == FakeHost + NightlyTagAsset)
                return Text(StableBuild + "\n");

            if (url.EndsWith("/releases/tags/v0.4.0", StringComparison.Ordinal))
                return Json(Release("v0.4.0", prerelease: false, [NightlyTagAsset]));

            foreach (var build in new[] { StableBuild, NewestNightly, OlderNightly })
            {
                if (url.EndsWith($"/releases/tags/{build}", StringComparison.Ordinal))
                    return Json(Release(build, prerelease: true, [CpuAssetFor(build)]));
                if (url == FakeHost + CpuAssetFor(build))
                    return Bytes(FakeReleaseAssets.ServerArchive());
            }

            if (url.Contains("/releases?", StringComparison.Ordinal))
                return Json("[" + string.Join(",",
                    Release("v0.5.0", prerelease: false, []),
                    Release(NewestNightly, prerelease: true, [CpuAssetFor(NewestNightly)]),
                    Release(OlderNightly, prerelease: true, [CpuAssetFor(OlderNightly)])) + "]");

            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
        }

        private static string Release(string tag, bool prerelease, string[] assets)
        {
            var assetJson = string.Join(",", assets.Select(a =>
                $$"""{ "name": "{{a}}", "browser_download_url": "{{FakeHost}}{{a}}", "size": 42 }"""));
            return $$"""{ "tag_name": "{{tag}}", "prerelease": {{(prerelease ? "true" : "false")}}, "assets": [{{assetJson}}] }""";
        }

        private static Task<HttpResponseMessage> Json(string json) =>
            Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(json, System.Text.Encoding.UTF8, "application/json") });

        private static Task<HttpResponseMessage> Text(string text) =>
            Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(text) });

        private static Task<HttpResponseMessage> Bytes(byte[] bytes) =>
            Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(bytes) });
    }
}
