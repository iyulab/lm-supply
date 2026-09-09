using System.Net;
using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// What the failure says when llama-server cannot be acquired.
///
/// <para>
/// Reported by a consumer whose machine had no GPU: local generation failed permanently with
/// <c>"No llama-server binary found for platform ..., backend Cpu"</c>, and the word <c>Cpu</c> in it was
/// read as "there is no CPU build" — a CPU-backend gap that does not exist. The real cause was upstream
/// release-tag resolution, and <c>Cpu</c> was only the last link of the fallback chain, printed as if it
/// had been the request. The message named neither the backend actually asked for, nor the release it
/// searched, nor what that release did contain, so nothing in it could correct the misreading.
/// </para>
/// <para>
/// These tests pin the diagnosis, not the prose: each asserts a fact the message must carry.
/// </para>
/// </summary>
public sealed class LlamaServerAcquisitionDiagnosticsTests : IDisposable
{
    private const string Build = "b10809";

    private readonly string _dir = Path.Combine(
        Path.GetTempPath(), "lmsupply-diag-" + Guid.NewGuid().ToString("N"));

    public LlamaServerAcquisitionDiagnosticsTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort temp cleanup */ }
        GC.SuppressFinalize(this);
    }

    private LlamaServerDownloader CreateDownloader(HttpMessageHandler handler) =>
        new(_dir, new HttpClient(handler));

    [Fact]
    public async Task WhenNoAssetMatches_TheFailureNamesTheBackendAskedFor_AndTheFallbackChainTried()
    {
        // A real release, but every asset in it is for some other platform.
        using var handler = new UnmatchableAssetsHandler(Build);
        var downloader = CreateDownloader(handler);

        var act = async () => await downloader.EnsureServerAsync(
            version: Build,
            preferredBackend: LlamaServerBackend.Cuda12,
            cancellationToken: TestContext.Current.CancellationToken);

        var message = (await act.Should().ThrowAsync<InvalidOperationException>()).Which.Message;

        message.Should().Contain("Cuda12",
            "the backend the caller asked for is the first thing a reader needs; printing only the last link of " +
            "the fallback chain is what made a resolution failure look like a missing CPU build");
        message.Should().Contain("Cpu",
            "the fallback chain that was actually walked belongs in the message too — but as a step, not as the request");
    }

    [Fact]
    public async Task WhenNoAssetMatches_TheFailureNamesTheReleaseSearched_AndWhatItContained()
    {
        using var handler = new UnmatchableAssetsHandler(Build);
        var downloader = CreateDownloader(handler);

        var act = async () => await downloader.EnsureServerAsync(
            version: Build,
            preferredBackend: LlamaServerBackend.Cpu,
            cancellationToken: TestContext.Current.CancellationToken);

        var message = (await act.Should().ThrowAsync<InvalidOperationException>()).Which.Message;

        message.Should().Contain(Build,
            "which release was searched is the difference between 'this build has no asset for me' and " +
            "'no release could be resolved at all'");
        message.Should().Contain(UnmatchableAssetsHandler.DecoyAsset,
            "listing what the release did contain is what lets a reader see the mismatch instead of guessing at it");
    }

    [Fact]
    public async Task WhenNoReleaseResolves_TheFailureSaysSo_RatherThanBlamingTheBackend()
    {
        using var handler = new NoReleasesHandler();
        var downloader = CreateDownloader(handler);

        var act = async () => await downloader.EnsureServerAsync(
            cancellationToken: TestContext.Current.CancellationToken);

        var message = (await act.Should().ThrowAsync<InvalidOperationException>()).Which.Message;

        message.Should().Contain("release",
            "nothing was searched, so naming a backend as the reason is a false lead — this is the exact " +
            "confusion the report describes");
    }

    // ---------------------------------------------------------------- fakes

    private sealed class UnmatchableAssetsHandler(string build) : HttpMessageHandler
    {
        public const string DecoyAsset = "llama-b10809-bin-freebsd-sparc.tar.gz";

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var url = request.RequestUri!.ToString();
            if (url.EndsWith($"/releases/tags/{build}", StringComparison.Ordinal))
            {
                var json = $$"""
                    { "tag_name": "{{build}}", "prerelease": true, "assets": [
                      { "name": "{{DecoyAsset}}", "browser_download_url": "https://fake.local/{{DecoyAsset}}", "size": 42 }
                    ] }
                    """;
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent(json, System.Text.Encoding.UTF8, "application/json"),
                });
            }

            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
        }
    }

    private sealed class NoReleasesHandler : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            => Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
    }
}
