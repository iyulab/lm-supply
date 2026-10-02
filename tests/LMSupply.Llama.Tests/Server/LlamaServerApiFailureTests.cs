using System.Net;
using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// A release lookup that GitHub refuses has to say why. GitHub allows an unauthenticated client 60 API requests an
/// hour per IP address; a busy machine or a shared address runs out, and the failure used to read "the latest-release
/// lookup returned nothing" with the cause only in the trace. Fully network-free (fake <see cref="HttpMessageHandler"/>).
/// </summary>
public sealed class LlamaServerApiFailureTests : IDisposable
{
    private const string Build = "b10809";
    private const long ResetEpoch = 1790924619; // 2026-10-02 07:03:39 UTC

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "lmsupply-apifail-" + Guid.NewGuid().ToString("N"));

    public LlamaServerApiFailureTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort temp cleanup */ }
        GC.SuppressFinalize(this);
    }

    [Fact]
    public async Task AnExhaustedRateLimit_IsNamed_WithTheResetTime_AndHowToRaiseIt()
    {
        using var http = new HttpClient(new RateLimitedHandler());
        using var downloader = new LlamaServerDownloader(_dir, http, includePrerelease: false, apiTimeout: null, apiToken: null);

        var resolution = await downloader.ResolveAssetAsync(null, LlamaServerBackend.Cpu, TestContext.Current.CancellationToken);

        resolution.Asset.Should().BeNull();
        resolution.Reason.Should().Be(LlamaServerAcquisitionFailure.ReleaseNotResolved);
        resolution.Failure.Should().Contain("rate limit exhausted")
            .And.Contain("unauthenticated")
            .And.Contain("60 requests per hour")
            .And.Contain("2026-10-02 07:03:39 UTC")
            .And.Contain("GITHUB_TOKEN")
            .And.NotContain("returned nothing");
    }

    [Fact]
    public async Task WithAToken_TheLimitIsTheToken_AndNoTokenHintIsGiven()
    {
        using var http = new HttpClient(new RateLimitedHandler());
        using var downloader = new LlamaServerDownloader(_dir, http, includePrerelease: false, apiTimeout: null, apiToken: "t0k");

        var resolution = await downloader.ResolveAssetAsync(null, LlamaServerBackend.Cpu, TestContext.Current.CancellationToken);

        resolution.Failure.Should().Contain("rate limit exhausted for this token").And.NotContain("Set GITHUB_TOKEN");
    }

    [Fact]
    public async Task AnyOtherRefusal_NamesTheStatus()
    {
        using var http = new HttpClient(new StatusHandler(HttpStatusCode.ServiceUnavailable));
        using var downloader = new LlamaServerDownloader(_dir, http, includePrerelease: false, apiTimeout: null, apiToken: null);

        var resolution = await downloader.ResolveAssetAsync(null, LlamaServerBackend.Cpu, TestContext.Current.CancellationToken);

        resolution.Failure.Should().Contain("503").And.Contain("/releases/latest");
    }

    [Fact]
    public async Task TheToken_GoesToTheApi_AndNeverToTheBuildDownload()
    {
        var handler = new RecordingReleasesHandler();
        using var http = new HttpClient(handler);
        using var downloader = new LlamaServerDownloader(_dir, http, includePrerelease: false, apiTimeout: null, apiToken: "t0k");

        var resolution = await downloader.ResolveAssetAsync(null, LlamaServerBackend.Cpu, TestContext.Current.CancellationToken);
        resolution.Asset.Should().NotBeNull(resolution.Failure);
        await downloader.DownloadAsync(resolution.Asset!, cancellationToken: TestContext.Current.CancellationToken);

        handler.Requests.Should().Contain(r => r.Url.Contains("api.github.com", StringComparison.Ordinal) && r.Authorization == "Bearer t0k");
        handler.Requests.Where(r => !r.Url.Contains("api.github.com", StringComparison.Ordinal))
            .Should().NotBeEmpty().And.OnlyContain(r => r.Authorization == null, "the build archive lives on another host");
    }

    [Fact]
    public async Task AClientWithItsOwnAuthorization_IsLeftAsItIs()
    {
        var handler = new RecordingReleasesHandler();
        using var http = new HttpClient(handler);
        http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("token", "mine");
        using var downloader = new LlamaServerDownloader(_dir, http, includePrerelease: false, apiTimeout: null, apiToken: "t0k");

        await downloader.ResolveAssetAsync(null, LlamaServerBackend.Cpu, TestContext.Current.CancellationToken);

        handler.Requests.Should().NotBeEmpty().And.OnlyContain(r => r.Authorization == "token mine");
    }

    private sealed class RateLimitedHandler : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var response = new HttpResponseMessage(HttpStatusCode.Forbidden) { RequestMessage = request };
            response.Headers.Add("x-ratelimit-limit", request.Headers.Authorization is null ? "60" : "5000");
            response.Headers.Add("x-ratelimit-remaining", "0");
            response.Headers.Add("x-ratelimit-reset", ResetEpoch.ToString(System.Globalization.CultureInfo.InvariantCulture));
            return Task.FromResult(response);
        }
    }

    private sealed class StatusHandler(HttpStatusCode status) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            => Task.FromResult(new HttpResponseMessage(status) { RequestMessage = request });
    }

    private sealed class RecordingReleasesHandler : HttpMessageHandler
    {
        private const string AssetHost = "https://objects.example/";

        public List<(string Url, string? Authorization)> Requests { get; } = [];

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var url = request.RequestUri!.ToString();
            var authorization = request.Headers.Authorization?.ToString()
                                ?? (request.Headers.TryGetValues("Authorization", out var values) ? values.First() : null);
            Requests.Add((url, authorization));

            var asset = FakeReleaseAssets.CpuAssetName(Build);
            if (url.EndsWith("/releases/latest", StringComparison.Ordinal) || url.EndsWith($"/releases/tags/{Build}", StringComparison.Ordinal))
            {
                var json = $$"""{ "tag_name": "{{Build}}", "prerelease": false, "assets": [{ "name": "{{asset}}", "browser_download_url": "{{AssetHost}}{{asset}}", "size": {{FakeReleaseAssets.ServerArchiveSize}} }] }""";
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent(json, System.Text.Encoding.UTF8, "application/json"),
                });
            }

            if (url == AssetHost + asset)
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(FakeReleaseAssets.ServerArchive()) });

            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound) { RequestMessage = request });
        }
    }
}
