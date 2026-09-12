using System.Reflection;
using AwesomeAssertions;
using LMSupply.Exceptions;
using LMSupply.Generator.Internal.Llama;
using LMSupply.Llama.Server;

namespace LMSupply.Generator.Tests;

/// <summary>
/// <see cref="GeneratorOptions.DisableAutoDownload"/> makes a load fail-closed on the GGUF paths (registry
/// alias, repository id) and on the ONNX download path: the cache is read, never written, no repository is
/// listed, and the refusal comes before any runtime binary (llama-server, GenAI) would be fetched.
/// </summary>
public sealed class OfflineLoadTests : IDisposable
{
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-generator-cache-" + Guid.NewGuid().ToString("N"));

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    [Fact]
    public void DisableAutoDownload_DefaultsToFalse() =>
        new GeneratorOptions().DisableAutoDownload.Should().BeFalse();

    [Fact]
    public async Task DisableAutoDownload_WithAGgufAliasNotCached_FailsTheLoadInsteadOfDownloading()
    {
        var alias = GgufModelRegistry.GetAliases()[0];
        var options = new GeneratorOptions { CacheDirectory = _cacheDir, DisableAutoDownload = true };

        var load = () => LocalGenerator.LoadAsync(alias, options, cancellationToken: Ct);

        await load.Should().ThrowAsync<ModelNotFoundException>().WithMessage("*downloads are disabled*");
        Directory.Exists(_cacheDir).Should().BeFalse("an offline miss downloads nothing and writes nothing to the cache");
    }

    [Fact]
    public async Task DisableAutoDownload_WithAGgufRepoIdNotCached_FailsTheLoadInsteadOfListingTheRepository()
    {
        var options = new GeneratorOptions { CacheDirectory = _cacheDir, DisableAutoDownload = true };

        var load = () => LocalGenerator.LoadAsync("bartowski/Qwen2.5-0.5B-Instruct-GGUF", options, cancellationToken: Ct);

        await load.Should().ThrowAsync<ModelNotFoundException>().WithMessage("*downloads are disabled*");
        Directory.Exists(_cacheDir).Should().BeFalse();
    }

    // The ONNX download half (shared by LoadAsync and DownloadModelAsync) refuses the same way.
    [Fact]
    public async Task DisableAutoDownload_WithAnOnnxRepoIdNotCached_FailsTheDownloadInsteadOfDiscovering()
    {
        var options = new GeneratorOptions { CacheDirectory = _cacheDir, DisableAutoDownload = true };

        var download = () => LocalGenerator.DownloadModelAsync("microsoft/Phi-3.5-mini-instruct-onnx", options, cancellationToken: Ct);

        await download.Should().ThrowAsync<ModelNotFoundException>();
        Directory.Exists(_cacheDir).Should().BeFalse();
    }

    // A cached GGUF file is what an offline load opens — the file, not a repository listing, decides.
    [Fact]
    public async Task DisableAutoDownload_WithAGgufFileCached_ResolvesThatFileWithoutListingTheRepository()
    {
        const string repoId = "example/some-model-GGUF";
        using var downloader = new GgufModelDownloader(_cacheDir, localFilesOnly: true);
        var seeded = Path.Combine(_cacheDir, "models--example--some-model-GGUF", "snapshots", "main", "some-model-Q4_K_M.gguf");
        Directory.CreateDirectory(Path.GetDirectoryName(seeded)!);
        await File.WriteAllTextAsync(seeded, "placeholder", Ct);

        var resolved = await downloader.DownloadAsync(repoId, cancellationToken: Ct);

        resolved.Should().Be(seeded);
    }

    // Every settable option, set to a non-default value, must survive Clone — the ONNX factory used to
    // rebuild the options by hand when applying its default provider and dropped seven of them.
    [Fact]
    public void Clone_CarriesEverySettableOption()
    {
        var props = typeof(GeneratorOptions).GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Where(p => p.SetMethod != null)
            .ToList();
        var source = new GeneratorOptions();
        foreach (var prop in props)
            prop.SetValue(source, NonDefault(prop.PropertyType, prop.GetValue(source)));

        var clone = source.Clone();

        foreach (var prop in props)
        {
            prop.GetValue(clone).Should().Be(prop.GetValue(source),
                $"GeneratorOptions.{prop.Name} must survive Clone — add it to the copy");
        }
    }

    private static object NonDefault(Type type, object? current)
    {
        var t = Nullable.GetUnderlyingType(type) ?? type;
        if (t.IsEnum)
            return Enum.GetValues(t).Cast<object>().First(v => !Equals(v, current) && !Equals(v, Activator.CreateInstance(t)));
        if (t == typeof(int)) return Equals(current, 7) ? 8 : 7;
        if (t == typeof(bool)) return current is true ? false : true;
        if (t == typeof(string)) return Equals(current, "probe") ? "probe-2" : "probe";
        if (t == typeof(LlamaOptions)) return new LlamaOptions();
        if (t == typeof(LlamaServerUpdateOptions)) return new LlamaServerUpdateOptions();
        throw new NotSupportedException($"No probe value for {type} — extend NonDefault when an option of that type is added.");
    }
}
