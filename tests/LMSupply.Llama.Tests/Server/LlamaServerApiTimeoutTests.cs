using System.Diagnostics;
using System.Net;
using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// <see cref="LlamaServerUpdateOptions.ApiTimeout"/> (default 10 s) was documented and never passed on:
/// the downloader used an HTTP client with the framework's 100-second default, so a GitHub API call that
/// hung held acquisition for that long. The timeout now bounds each API request, and only those — a
/// build download must not share a ten-second limit. Network-free (fake handler).
/// </summary>
public sealed class LlamaServerApiTimeoutTests : IDisposable
{
    private const string Build = "b100";
    private const string FakeHost = "https://fake.local/";

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "lmsupply-apitimeout-" + Guid.NewGuid().ToString("N"));

    public LlamaServerApiTimeoutTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort temp cleanup */ }
    }

    [Fact]
    public async Task AHungApiRequest_EndsAtTheApiTimeout()
    {
        var handler = new Hub(apiHangs: true);
        using var downloader = new LlamaServerDownloader(_dir, new HttpClient(handler), apiTimeout: TimeSpan.FromMilliseconds(200));
        using var guard = Guard();
        var clock = Stopwatch.StartNew();

        var latest = await downloader.GetLatestVersionAsync(guard.Token);

        latest.Should().BeNull("a lookup that ran out of time counts as GitHub being unreachable");
        clock.Elapsed.Should().BeLessThan(TimeSpan.FromSeconds(5));
        handler.ApiRequestCancelled.Should().BeTrue("the request itself must be abandoned, not left running");
    }

    [Fact]
    public async Task TheServiceHandsTheOptionToItsDownloader()
    {
        var handler = new Hub(apiHangs: true);
        using var http = new HttpClient(handler);
        var options = new LlamaServerUpdateOptions { CacheDirectory = _dir, ApiTimeout = TimeSpan.FromMilliseconds(200) };
        await using var service = new LlamaServerUpdateService(options, http);
        using var guard = Guard();
        var clock = Stopwatch.StartNew();

        // Nothing is cached and GitHub did not answer in time: acquisition fails the way it does when
        // GitHub is unreachable — and it fails after the configured 200 ms, not the HTTP client's 100 s.
        var act = () => service.GetServerPathAsync(LlamaServerBackend.Cpu, cancellationToken: guard.Token);

        await act.Should().ThrowAsync<LlamaServerAcquisitionException>();
        clock.Elapsed.Should().BeLessThan(TimeSpan.FromSeconds(5));
    }

    [Fact]
    public async Task ASlowBuildDownload_IsNotCutOffByTheApiTimeout()
    {
        var handler = new Hub(apiHangs: false, assetDelay: TimeSpan.FromMilliseconds(600));
        using var http = new HttpClient(handler);
        var options = new LlamaServerUpdateOptions { CacheDirectory = _dir, ApiTimeout = TimeSpan.FromMilliseconds(150) };
        await using var service = new LlamaServerUpdateService(options, http);
        using var guard = Guard();

        var result = await service.GetServerPathAsync(LlamaServerBackend.Cpu, cancellationToken: guard.Token);

        result.Success.Should().BeTrue(result.Error);
        result.NewVersion.Should().Be(Build);
    }

    [Fact]
    public void ANonPositiveTimeout_IsRejected()
    {
        var act = () => new LlamaServerDownloader(_dir, new HttpClient(new Hub(apiHangs: false)), apiTimeout: TimeSpan.Zero);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    // Without the fix the hung request would wait for the HTTP client's 100 seconds; this bounds the test.
    private static CancellationTokenSource Guard()
    {
        var cts = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        cts.CancelAfter(TimeSpan.FromSeconds(10));
        return cts;
    }

    /// <summary>
    /// A one-build GitHub: <c>releases/latest</c> and <c>releases/tags/b100</c> name this platform's CPU
    /// asset. API requests either answer at once or hang until cancelled; the asset answers after a delay.
    /// </summary>
    private sealed class Hub(bool apiHangs, TimeSpan assetDelay = default) : HttpMessageHandler
    {
        public bool ApiRequestCancelled { get; private set; }

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var url = request.RequestUri!.ToString();
            var asset = FakeReleaseAssets.CpuAssetName(Build);

            if (url == FakeHost + asset)
            {
                await Task.Delay(assetDelay, cancellationToken);
                return new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(FakeReleaseAssets.ServerArchive()) };
            }

            if (url.Contains("/releases", StringComparison.Ordinal))
            {
                if (apiHangs)
                {
                    try
                    {
                        await Task.Delay(Timeout.Infinite, cancellationToken);
                    }
                    catch (OperationCanceledException)
                    {
                        ApiRequestCancelled = true;
                        throw;
                    }
                }

                var json = $$"""{ "tag_name": "{{Build}}", "prerelease": false, "assets": [{ "name": "{{asset}}", "browser_download_url": "{{FakeHost}}{{asset}}", "size": 42 }] }""";
                return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(json, System.Text.Encoding.UTF8, "application/json") };
            }

            return new HttpResponseMessage(HttpStatusCode.NotFound);
        }
    }
}
