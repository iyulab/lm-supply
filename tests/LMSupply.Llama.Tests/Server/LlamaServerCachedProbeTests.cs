using AwesomeAssertions;
using LMSupply.Llama.Server;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// <see cref="LlamaServerDownloader.IsAnyServerCached"/> answers "is a server binary already here" from
/// the cache layout alone (<c>&lt;cache&gt;/&lt;build tag&gt;/&lt;backend&gt;/llama-server[.exe]</c>) — no
/// request, no version resolution — so a caller can choose the GGUF route without ever causing a
/// server download.
/// </summary>
public sealed class LlamaServerCachedProbeTests : IDisposable
{
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), "lmsupply-server-probe-" + Guid.NewGuid().ToString("N"));
    private static string Executable => OperatingSystem.IsWindows() ? "llama-server.exe" : "llama-server";

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir))
            Directory.Delete(_cacheDir, recursive: true);
    }

    [Fact]
    public void A_missing_or_empty_cache_is_not_cached()
    {
        LlamaServerDownloader.IsAnyServerCached(_cacheDir).Should().BeFalse("the directory does not exist");
        Directory.CreateDirectory(_cacheDir);
        LlamaServerDownloader.IsAnyServerCached(_cacheDir).Should().BeFalse("the directory is empty");
    }

    [Fact]
    public async Task A_build_with_an_executable_for_any_backend_is_cached()
    {
        var dir = Path.Combine(_cacheDir, "b10868", "cpu");
        Directory.CreateDirectory(dir);
        await File.WriteAllTextAsync(Path.Combine(dir, Executable), "binary", TestContext.Current.CancellationToken);

        LlamaServerDownloader.IsAnyServerCached(_cacheDir).Should().BeTrue();
    }

    [Fact]
    public async Task A_directory_that_is_not_a_build_tag_and_a_build_without_the_executable_do_not_count()
    {
        Directory.CreateDirectory(Path.Combine(_cacheDir, "v0.4.0", "cpu"));
        await File.WriteAllTextAsync(Path.Combine(_cacheDir, "v0.4.0", "cpu", Executable), "binary", TestContext.Current.CancellationToken);
        Directory.CreateDirectory(Path.Combine(_cacheDir, "b10868", "vulkan"));   // extracted nothing

        LlamaServerDownloader.IsAnyServerCached(_cacheDir).Should().BeFalse();
    }
}
