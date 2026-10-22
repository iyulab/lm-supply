using System.Net;
using AwesomeAssertions;
using LMSupply.Exceptions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// A llama-server build that is not cached and whose source cannot be reached at all fails with
/// <see cref="ModelHostUnreachableException"/> naming that host — the same failure a model download raises — so a
/// host can tell an unreachable source from a release that has no build for it. Network-free (fake handlers that
/// refuse the connection).
/// </summary>
public sealed class LlamaServerHostUnreachableTests : IDisposable
{
    private const string Build = "b9100";
    private const string FakeHost = "https://fake.local/";

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "lmsupply-host-unreachable-" + Guid.NewGuid().ToString("N"));

    public LlamaServerHostUnreachableTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort temp cleanup */ }
    }

    [Fact]
    public async Task AnUnreachableReleaseApi_IsTypedWithItsHost()
    {
        using var downloader = new LlamaServerDownloader(_dir, new HttpClient(new Refusing(refuseApi: true)));

        var act = () => downloader.EnsureServerAsync(preferredBackend: LlamaServerBackend.Cpu, cancellationToken: TestContext.Current.CancellationToken);

        var failure = (await act.Should().ThrowAsync<ModelHostUnreachableException>()).Which;
        failure.Host.Should().Be("api.github.com");
        failure.InnerException.Should().BeOfType<HttpRequestException>();
    }

    [Fact]
    public async Task AnUnreachableReleaseApi_ForANamedRelease_IsTypedWithItsHost()
    {
        using var downloader = new LlamaServerDownloader(_dir, new HttpClient(new Refusing(refuseApi: true)));

        var act = () => downloader.EnsureServerAsync(version: "v1.2.3", preferredBackend: LlamaServerBackend.Cpu, cancellationToken: TestContext.Current.CancellationToken);

        (await act.Should().ThrowAsync<ModelHostUnreachableException>()).Which.Host.Should().Be("api.github.com");
    }

    [Fact]
    public async Task AnUnreachableBuildDownload_IsTypedWithTheDownloadHost()
    {
        using var downloader = new LlamaServerDownloader(_dir, new HttpClient(new Refusing(refuseApi: false)));

        var act = () => downloader.EnsureServerAsync(version: Build, preferredBackend: LlamaServerBackend.Cpu, cancellationToken: TestContext.Current.CancellationToken);

        (await act.Should().ThrowAsync<ModelHostUnreachableException>()).Which.Host.Should().Be("fake.local");
    }

    [Fact]
    public async Task TheFirstLoadThroughTheService_IsTypedToo()
    {
        using var http = new HttpClient(new Refusing(refuseApi: true));
        await using var service = new LlamaServerUpdateService(new LlamaServerUpdateOptions { CacheDirectory = _dir }, http);

        var act = () => service.GetServerPathAsync(LlamaServerBackend.Cpu, cancellationToken: TestContext.Current.CancellationToken);

        (await act.Should().ThrowAsync<ModelHostUnreachableException>()).Which.Host.Should().Be("api.github.com");
    }

    [Fact]
    public async Task AReleaseApiThatAnswers_StaysAnAcquisitionFailure()
    {
        // GitHub answered (404): it is reachable, so this is not an unreachable host.
        using var downloader = new LlamaServerDownloader(_dir, new HttpClient(new NotFound()));

        var act = () => downloader.EnsureServerAsync(preferredBackend: LlamaServerBackend.Cpu, cancellationToken: TestContext.Current.CancellationToken);

        (await act.Should().ThrowAsync<LlamaServerAcquisitionException>()).Which.Reason.Should().Be(LlamaServerAcquisitionFailure.ReleaseNotResolved);
    }

    /// <summary>Refuses the connection to the GitHub API, or answers it with one build and refuses the build download.</summary>
    private sealed class Refusing(bool refuseApi) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var url = request.RequestUri!.ToString();
            if (url.Contains("/releases", StringComparison.Ordinal) && !refuseApi)
            {
                var asset = FakeReleaseAssets.CpuAssetName(Build);
                var json = $$"""{ "tag_name": "{{Build}}", "prerelease": false, "assets": [{ "name": "{{asset}}", "browser_download_url": "{{FakeHost}}{{asset}}", "size": {{FakeReleaseAssets.ServerArchiveSize}} }] }""";
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(json, System.Text.Encoding.UTF8, "application/json") });
            }

            throw new HttpRequestException(HttpRequestError.ConnectionError, "No connection could be made because the target machine actively refused it.");
        }
    }

    private sealed class NotFound : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            => Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
    }
}
