using AwesomeAssertions;
using LMSupply.Runtime;

namespace LMSupply.Core.Tests.Runtime;

/// <summary>
/// Guards the single-root contract for runtime-package caching (Forge upstream-019):
/// the runtime family (RuntimeManager, RuntimeVersionStateManager, OnnxNuGetDownloader)
/// must resolve its default directory through <see cref="LMSupplyCachePaths"/> —
/// LMSUPPLY_CACHE_DIR env → %LOCALAPPDATA%/LMSupply/cache/runtimes — never through
/// hardcoded copies, and never nested inside a HuggingFace hub directory.
///
/// Shares the "lmsupply-cache-env" collection with every other class mutating the
/// process-global LMSUPPLY_CACHE_DIR env var (see LMSupplyCachePathsTests) — parallel
/// classes would otherwise clobber each other's env state.
/// </summary>
[Collection("lmsupply-cache-env")]
public sealed class RuntimeCachePathUnificationTests
{
    [Fact]
    public void RuntimeVersionStateManager_Default_HonorsLmSupplyCacheDirEnv()
    {
        var original = Environment.GetEnvironmentVariable(LMSupplyCachePaths.RootEnvironmentVariable);
        var custom = Path.Combine(Path.GetTempPath(), "lmsupply-tests", "runtime-root");
        try
        {
            Environment.SetEnvironmentVariable(LMSupplyCachePaths.RootEnvironmentVariable, custom);

            using var manager = new RuntimeVersionStateManager();

            manager.StateFilePath.Should().Be(
                Path.Combine(custom, "runtimes", "runtime-versions.json"),
                "the default state path must resolve through the unified LMSupply root");
        }
        finally
        {
            Environment.SetEnvironmentVariable(LMSupplyCachePaths.RootEnvironmentVariable, original);
        }
    }

    [Fact]
    public void RuntimeManager_DefaultCacheDirectory_HonorsLmSupplyCacheDirEnv()
    {
        var original = Environment.GetEnvironmentVariable(LMSupplyCachePaths.RootEnvironmentVariable);
        var custom = Path.Combine(Path.GetTempPath(), "lmsupply-tests", "runtime-mgr-root");
        try
        {
            Environment.SetEnvironmentVariable(LMSupplyCachePaths.RootEnvironmentVariable, custom);

            var manager = new RuntimeManager(new RuntimeManagerOptions());

            manager.CacheDirectory.Should().Be(Path.Combine(custom, "runtimes"));
        }
        finally
        {
            Environment.SetEnvironmentVariable(LMSupplyCachePaths.RootEnvironmentVariable, original);
        }
    }

    [Fact]
    public void RuntimeManager_DefaultCacheDirectory_MatchesUnifiedResolver()
    {
        var manager = new RuntimeManager(new RuntimeManagerOptions());

        manager.CacheDirectory.Should().Be(LMSupplyCachePaths.GetRuntimesDirectory(),
            "whatever the environment, the runtime family and the unified resolver must agree");
    }
}
