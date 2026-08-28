using AwesomeAssertions;
using LMSupply.Generator.Internal.Llama;

namespace LMSupply.Generator.Tests;

/// <summary>
/// GGUF downloads authenticate with the same <c>HF_TOKEN</c> the ONNX path uses.
///
/// <para>
/// They did not until 2026-08-06, and the asymmetry was invisible from the outside: unauthenticated
/// requests to HuggingFace succeed until a rate limit is reached, so nothing failed until a cold
/// pull on a shared IP got 429 — at which point no cache directory had been created either, so the
/// next attempt started from the same place. Meanwhile the sibling path's own error text instructed
/// the caller to "Set HF_TOKEN environment variable", advice that did nothing for the large files.
/// </para>
///
/// <para>
/// These tests are about WHERE the token comes from, not about talking to HuggingFace. One source
/// for both paths is the property that matters: a second environment variable, or a token honoured
/// on one path only, is how this defect existed in the first place.
/// </para>
/// </summary>
public class GgufDownloadAuthenticationTests : IDisposable
{
    private readonly string? _originalToken = Environment.GetEnvironmentVariable("HF_TOKEN");

    public void Dispose()
    {
        Environment.SetEnvironmentVariable("HF_TOKEN", _originalToken);
        GC.SuppressFinalize(this);
    }

    [Fact]
    public void ExplicitToken_IsSent()
    {
        Environment.SetEnvironmentVariable("HF_TOKEN", null);

        using var downloader = new GgufModelDownloader(cacheDirectory: null, hfToken: "explicit-token");

        downloader.AuthorizationScheme.Should().Be("Bearer");
    }

    [Fact]
    public void EnvironmentToken_IsSent_WhenNoneIsPassedExplicitly()
    {
        // The same variable the ONNX path reads. A different name here would leave a consumer who
        // followed the documented advice still unauthenticated on the path that needs it most.
        Environment.SetEnvironmentVariable("HF_TOKEN", "env-token");

        using var downloader = new GgufModelDownloader(cacheDirectory: null);

        downloader.AuthorizationScheme.Should().Be("Bearer");
    }

    [Fact]
    public void ExplicitToken_TakesPrecedenceOverTheEnvironment()
    {
        Environment.SetEnvironmentVariable("HF_TOKEN", "env-token");

        using var downloader = new GgufModelDownloader(cacheDirectory: null, hfToken: "explicit-token");

        downloader.AuthorizationScheme.Should().Be("Bearer");
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    public void NoToken_SendsNoCredentials(string? token)
    {
        // Anonymous access stays the default. Sending an empty Bearer header would turn a working
        // anonymous download into a 401.
        Environment.SetEnvironmentVariable("HF_TOKEN", token);

        using var downloader = new GgufModelDownloader(cacheDirectory: null);

        downloader.AuthorizationScheme.Should().BeNull();
    }
}
