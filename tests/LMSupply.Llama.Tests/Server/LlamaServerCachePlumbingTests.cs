using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// Guards the override chain for the llama-server binary cache (Forge upstream-019 P1):
/// before this, <see cref="LlamaServerUpdateService"/> constructed its downloader and state
/// manager arg-less against a hardcoded path, so no caller — options, env, anything — could
/// relocate or inspect the binary store. The chain is now:
/// explicit ctor arg → LlamaServerUpdateOptions.CacheDirectory → LMSUPPLY_CACHE_DIR env →
/// %LOCALAPPDATA%/LMSupply/cache/llama-server.
///
/// Env mutation notes: only these new resolver paths read LMSUPPLY_CACHE_DIR; all other tests
/// pass explicit cache directories, so parallel execution cannot observe the mutation.
/// </summary>
public sealed class LlamaServerCachePlumbingTests : IDisposable
{
    private readonly string _root;

    public LlamaServerCachePlumbingTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "lmsupply-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { /* best effort */ }
    }

    [Fact]
    public void StateManager_Default_HonorsLmSupplyCacheDirEnv()
    {
        var original = Environment.GetEnvironmentVariable("LMSUPPLY_CACHE_DIR");
        try
        {
            Environment.SetEnvironmentVariable("LMSUPPLY_CACHE_DIR", _root);

            using var manager = new LlamaServerStateManager();

            Directory.Exists(Path.Combine(_root, "llama-server")).Should().BeTrue(
                "the arg-less state manager must land under the env-overridden LMSupply root");
        }
        finally
        {
            Environment.SetEnvironmentVariable("LMSUPPLY_CACHE_DIR", original);
        }
    }

    [Fact]
    public void Downloader_Default_HonorsLmSupplyCacheDirEnv()
    {
        var original = Environment.GetEnvironmentVariable("LMSUPPLY_CACHE_DIR");
        try
        {
            Environment.SetEnvironmentVariable("LMSUPPLY_CACHE_DIR", _root);

            var versionDir = Path.Combine(_root, "llama-server", "b123", "vulkan");
            Directory.CreateDirectory(versionDir);

            using var downloader = new LlamaServerDownloader();

            downloader.GetCachedVersions().Should().ContainSingle()
                .Which.Should().Be("b123", "the arg-less downloader must scan the env-overridden root");
        }
        finally
        {
            Environment.SetEnvironmentVariable("LMSUPPLY_CACHE_DIR", original);
        }
    }

    [Fact]
    public async Task UpdateService_PlumbsOptionsCacheDirectory_IntoItsComponents()
    {
        var cacheDir = Path.Combine(_root, "custom-llama-store");

        await using var service = new LlamaServerUpdateService(new LlamaServerUpdateOptions
        {
            CacheDirectory = cacheDir,
            AutoDownloadUpdates = false
        });

        Directory.Exists(cacheDir).Should().BeTrue(
            "the update service must construct its state manager against the configured directory");
        File.Exists(Path.Combine(cacheDir, "llama-server-state.json")).Should().BeFalse(
            "no state is written until an installation happens — only the directory is prepared");
    }
}
