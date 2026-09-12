using AwesomeAssertions;
using LMSupply.Exceptions;

namespace LMSupply.Synthesizer.Tests;

/// <summary>
/// <see cref="SynthesizerOptions.DisableAutoDownload"/> makes a load fail-closed: the cache is read, never
/// written, and a voice that is not there ends the load instead of starting a download. (The loader keeps a
/// <see cref="SynthesizerOptions.Clone"/> of the caller's options; that the clone carries every option is
/// pinned centrally by <c>OptionsCloneCompletenessTests</c> in the integration project.)
/// </summary>
public sealed class OfflineLoadTests : IDisposable
{
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-synthesizer-cache-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    [Fact]
    public void DisableAutoDownload_DefaultsToFalse() =>
        new SynthesizerOptions().DisableAutoDownload.Should().BeFalse();

    [Fact]
    public async Task DisableAutoDownload_WithTheVoiceNotCached_FailsTheLoadInsteadOfDownloading()
    {
        var options = new SynthesizerOptions { CacheDirectory = _cacheDir, DisableAutoDownload = true };

        var load = () => LocalSynthesizer.LoadAsync("default", options, cancellationToken: Ct);

        await load.Should().ThrowAsync<ModelNotFoundException>().WithMessage("*downloads are disabled*");
        Directory.Exists(_cacheDir).Should().BeFalse("an offline miss downloads nothing and writes nothing to the cache");
    }
}
