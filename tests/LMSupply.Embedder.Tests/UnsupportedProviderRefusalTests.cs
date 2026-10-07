using AwesomeAssertions;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// An explicit <c>ExecutionProvider</c> value 2 (DirectML, removed in 0.111.0) pin is refused at the entry point — before any
/// model resolution, download or tokenizer load. Until 0.67.1 the refusal lived in the session factory,
/// so a cache-miss load downloaded hundreds of megabytes and then threw.
/// </summary>
public sealed class UnsupportedProviderRefusalTests : IDisposable
{
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-embedder-refusal-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    [Theory]
    [InlineData("default")]
    [InlineData("BAAI/bge-m3")]
    public async Task ExplicitDirectML_IsRefusedBeforeAnyDownload(string modelId)
    {
        var options = new EmbedderOptions { CacheDirectory = _cacheDir, Provider = (ExecutionProvider)2 };

        var load = () => LocalEmbedder.LoadAsync(modelId, options, cancellationToken: TestContext.Current.CancellationToken);

        await load.Should().ThrowAsync<NotSupportedException>().WithMessage("*DirectML*");
        Directory.Exists(_cacheDir).Should().BeFalse("the provider is refused before anything is resolved or downloaded");
    }
}
