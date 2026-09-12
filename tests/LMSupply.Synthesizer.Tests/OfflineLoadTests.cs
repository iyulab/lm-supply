using System.Reflection;
using AwesomeAssertions;
using LMSupply.Exceptions;

namespace LMSupply.Synthesizer.Tests;

/// <summary>
/// <see cref="SynthesizerOptions.DisableAutoDownload"/> makes a load fail-closed: the cache is read, never
/// written, and a voice that is not there ends the load instead of starting a download. The loader keeps a
/// <see cref="SynthesizerOptions.Clone"/> of the caller's options, so the clone must carry every option —
/// until 0.65.0 it dropped <see cref="LMSupplyOptionsBase.LogLevel"/>, which made that option a no-op.
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

    // Every settable option, set to a non-default value, must survive Clone — a property added to the
    // type and forgotten in the copy is silently a no-op for every load (the loader only sees the clone).
    [Fact]
    public void Clone_CarriesEverySettableOption()
    {
        var props = typeof(SynthesizerOptions).GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Where(p => p.SetMethod != null)
            .ToList();
        var source = new SynthesizerOptions();
        foreach (var prop in props)
            prop.SetValue(source, NonDefault(prop.PropertyType, prop.GetValue(source)));

        var clone = source.Clone();

        foreach (var prop in props)
        {
            prop.GetValue(clone).Should().Be(prop.GetValue(source),
                $"SynthesizerOptions.{prop.Name} must survive Clone — add it to the copy");
        }
    }

    private static object NonDefault(Type type, object? current)
    {
        var t = Nullable.GetUnderlyingType(type) ?? type;
        if (t.IsEnum)
            return Enum.GetValues(t).Cast<object>().First(v => !Equals(v, current) && !Equals(v, Activator.CreateInstance(t)));
        if (t == typeof(int)) return Equals(current, 7) ? 8 : 7;
        if (t == typeof(float)) return Equals(current, 1.5f) ? 2.5f : 1.5f;
        if (t == typeof(bool)) return current is true ? false : true;
        if (t == typeof(string)) return Equals(current, "probe") ? "probe-2" : "probe";
        throw new NotSupportedException($"No probe value for {type} — extend NonDefault when an option of that type is added.");
    }
}
