using AwesomeAssertions;
using LMSupply.Exceptions;

namespace LMSupply.Generator.Onnx.Tests;

/// <summary>
/// <see cref="OnnxGeneratorModelFactory.LoadAsync"/> honours <see cref="GeneratorOptions.DisableAutoDownload"/>:
/// a model with no cached layout fails with <see cref="ModelNotFoundException"/> instead of being downloaded,
/// and the cache is left as it was.
/// </summary>
public sealed class OfflineLoadTests : IDisposable
{
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-genonnx-cache-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    [Fact]
    public async Task LoadAsync_WithDownloadsDisabledAndNothingCached_FailsInsteadOfDownloading()
    {
        using var factory = new OnnxGeneratorModelFactory(_cacheDir, ExecutionProvider.Cpu);
        var options = new GeneratorOptions { DisableAutoDownload = true };

        var load = () => factory.LoadAsync("microsoft/Phi-3.5-mini-instruct-onnx", options, TestContext.Current.CancellationToken);

        await load.Should().ThrowAsync<ModelNotFoundException>().WithMessage("*downloads are disabled*");
        Directory.Exists(_cacheDir).Should().BeFalse("an offline miss downloads nothing and writes nothing to the cache");
    }
}
