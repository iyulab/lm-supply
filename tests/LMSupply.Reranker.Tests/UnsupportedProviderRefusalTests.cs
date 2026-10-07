using AwesomeAssertions;

namespace LMSupply.Reranker.Tests;

/// <summary>
/// An explicit <c>ExecutionProvider</c> value 2 (DirectML, removed in 0.111.0) pin is refused at the entry point — before any
/// model resolution or download (0.67.1; see the embedder's twin test).
/// </summary>
public sealed class UnsupportedProviderRefusalTests : IDisposable
{
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-reranker-refusal-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    [Fact]
    public async Task ExplicitDirectML_IsRefusedBeforeAnyDownload()
    {
        var options = new RerankerOptions { CacheDirectory = _cacheDir, Provider = (ExecutionProvider)2 };

        var load = () => LocalReranker.LoadAsync("default", options, cancellationToken: TestContext.Current.CancellationToken);

        await load.Should().ThrowAsync<NotSupportedException>().WithMessage("*DirectML*");
        Directory.Exists(_cacheDir).Should().BeFalse("the provider is refused before anything is resolved or downloaded");
    }
}
