using System.Net;
using AwesomeAssertions;
using LMSupply.Exceptions;
using LMSupply.Generator.Gguf;

namespace LMSupply.Generator.Tests;

/// <summary>
/// A GGUF model that is not cached and whose hub cannot be reached at all fails with
/// <see cref="ModelHostUnreachableException"/> naming the hub — the failure the ONNX model path already raised.
/// The GGUF path goes through the hub cache and the shared resumable download rather than the ONNX downloader,
/// and surfaced the raw <see cref="HttpRequestException"/> until the shared download typed it. Network-free (a
/// handler that refuses every connection).
/// </summary>
public sealed class GgufHostUnreachableTests : IDisposable
{
    private readonly string _cache = Path.Combine(Path.GetTempPath(), "lmsupply-gguf-unreachable-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_cache))
            Directory.Delete(_cache, recursive: true);
    }

    [Fact]
    public async Task AnUncachedModel_WithTheHubUnreachable_IsTypedWithTheHubHost()
    {
        using var downloader = new GgufModelDownloader(_cache, hfToken: null, localFilesOnly: false, handler: new Refusing());

        var act = () => downloader.DownloadAsync(
            "example-org/example-model-GGUF", "example-model-Q4_K_M.gguf", cancellationToken: TestContext.Current.CancellationToken);

        var failure = (await act.Should().ThrowAsync<ModelHostUnreachableException>()).Which;
        var endpoint = Environment.GetEnvironmentVariable("HF_ENDPOINT");
        failure.Host.Should().Be(string.IsNullOrWhiteSpace(endpoint) ? "huggingface.co" : new Uri(endpoint).Host);
        failure.InnerException.Should().BeOfType<HttpRequestException>();
    }

    // A repository given without a file name is listed first to choose the file; that listing surfaced the raw
    // HttpRequestException after the file download was typed.
    [Fact]
    public async Task ARepositoryWithoutAFileName_WithTheHubUnreachable_IsTypedWithTheHubHost()
    {
        using var downloader = new GgufModelDownloader(_cache, hfToken: null, localFilesOnly: false, handler: new Refusing());

        var act = () => downloader.DownloadAsync("example-org/example-model-GGUF", cancellationToken: TestContext.Current.CancellationToken);

        var failure = (await act.Should().ThrowAsync<ModelHostUnreachableException>()).Which;
        var endpoint = Environment.GetEnvironmentVariable("HF_ENDPOINT");
        failure.Host.Should().Be(string.IsNullOrWhiteSpace(endpoint) ? "huggingface.co" : new Uri(endpoint).Host);
        failure.ModelId.Should().Be("example-org/example-model-GGUF");
        failure.InnerException.Should().BeOfType<HttpRequestException>();
    }

    [Fact]
    public async Task ARepositoryWithoutAFileName_WhoseHubAnswersWithAnError_IsNotAnUnreachableHost()
    {
        using var downloader = new GgufModelDownloader(_cache, hfToken: null, localFilesOnly: false, handler: new Status(HttpStatusCode.Forbidden));

        var act = () => downloader.DownloadAsync("example-org/example-model-GGUF", cancellationToken: TestContext.Current.CancellationToken);

        (await act.Should().ThrowAsync<Exception>()).Which.Should().NotBeOfType<ModelHostUnreachableException>();
    }

    [Fact]
    public async Task AHubThatAnswersWithAnError_IsNotAnUnreachableHost()
    {
        using var downloader = new GgufModelDownloader(_cache, hfToken: null, localFilesOnly: false, handler: new Status(HttpStatusCode.Forbidden));

        var act = () => downloader.DownloadAsync(
            "example-org/example-model-GGUF", "example-model-Q4_K_M.gguf", cancellationToken: TestContext.Current.CancellationToken);

        (await act.Should().ThrowAsync<Exception>()).Which.Should().NotBeOfType<ModelHostUnreachableException>();
    }

    private sealed class Refusing : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            => throw new HttpRequestException(HttpRequestError.ConnectionError, "No connection could be made because the target machine actively refused it.");
    }

    private sealed class Status(HttpStatusCode code) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            => Task.FromResult(new HttpResponseMessage(code));
    }
}
