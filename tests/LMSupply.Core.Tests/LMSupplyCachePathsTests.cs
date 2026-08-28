using AwesomeAssertions;

namespace LMSupply.Core.Tests;

/// <summary>
/// Tests for <see cref="LMSupplyCachePaths"/>, the single resolver for the LMSupply non-HF
/// artifact root (runtime packages, llama-server builds). Forge upstream-019 established that
/// these artifacts were resolved by several disagreeing hardcoded implementations with no
/// override chain; this resolver is the one place the root is decided:
/// LMSUPPLY_CACHE_DIR env → %LOCALAPPDATA%/LMSupply/cache.
///
/// HF model caching is deliberately separate (CacheManager, HF env chain) — models follow the
/// HuggingFace hub standard, non-HF artifacts must live outside any hub.
///
/// Collection note: every test class that mutates the process-global LMSUPPLY_CACHE_DIR env var
/// must share the "lmsupply-cache-env" collection — xUnit runs collections' classes sequentially,
/// preventing parallel classes from clobbering each other's env state.
/// </summary>
[Collection("lmsupply-cache-env")]
public sealed class LMSupplyCachePathsTests
{
    [Fact]
    public void RootDirectory_Default_IsLocalAppDataLMSupplyCache()
    {
        var original = Environment.GetEnvironmentVariable(LMSupplyCachePaths.RootEnvironmentVariable);
        try
        {
            Environment.SetEnvironmentVariable(LMSupplyCachePaths.RootEnvironmentVariable, null);

            var root = LMSupplyCachePaths.GetRootDirectory();

            root.Should().Be(Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "LMSupply", "cache"));
        }
        finally
        {
            Environment.SetEnvironmentVariable(LMSupplyCachePaths.RootEnvironmentVariable, original);
        }
    }

    [Fact]
    public void RootDirectory_HonorsEnvironmentOverride()
    {
        var original = Environment.GetEnvironmentVariable(LMSupplyCachePaths.RootEnvironmentVariable);
        var custom = Path.Combine(Path.GetTempPath(), "lmsupply-tests", "custom-root");
        try
        {
            Environment.SetEnvironmentVariable(LMSupplyCachePaths.RootEnvironmentVariable, custom);

            LMSupplyCachePaths.GetRootDirectory().Should().Be(custom);
        }
        finally
        {
            Environment.SetEnvironmentVariable(LMSupplyCachePaths.RootEnvironmentVariable, original);
        }
    }

    [Fact]
    public void ArtifactDirectories_LiveUnderTheRoot()
    {
        var original = Environment.GetEnvironmentVariable(LMSupplyCachePaths.RootEnvironmentVariable);
        var custom = Path.Combine(Path.GetTempPath(), "lmsupply-tests", "artifact-root");
        try
        {
            Environment.SetEnvironmentVariable(LMSupplyCachePaths.RootEnvironmentVariable, custom);

            LMSupplyCachePaths.GetRuntimesDirectory().Should().Be(Path.Combine(custom, "runtimes"));
            LMSupplyCachePaths.GetLlamaServerDirectory().Should().Be(Path.Combine(custom, "llama-server"));
        }
        finally
        {
            Environment.SetEnvironmentVariable(LMSupplyCachePaths.RootEnvironmentVariable, original);
        }
    }

    [Fact]
    public void RootDirectory_IgnoresWhitespaceOverride()
    {
        var original = Environment.GetEnvironmentVariable(LMSupplyCachePaths.RootEnvironmentVariable);
        try
        {
            Environment.SetEnvironmentVariable(LMSupplyCachePaths.RootEnvironmentVariable, "   ");

            LMSupplyCachePaths.GetRootDirectory().Should().Be(Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "LMSupply", "cache"));
        }
        finally
        {
            Environment.SetEnvironmentVariable(LMSupplyCachePaths.RootEnvironmentVariable, original);
        }
    }
}
