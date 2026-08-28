using AwesomeAssertions;
using LMSupply.Download;
using Xunit;

namespace LMSupply.Generator.Tests;

/// <summary>
/// Guards the single-resolver contract for the model cache (Forge upstream-019 P2): every
/// Generator code path must resolve the default cache directory through
/// <see cref="CacheManager.GetDefaultCacheDirectory"/> (HF env chain: HF_HUB_CACHE → HF_HOME →
/// XDG_CACHE_HOME → ~/.cache/huggingface/hub). Private hardcoded copies split the cache: a user
/// setting HF_HUB_CACHE got models resolved into two different hubs depending on which code path
/// ran, violating the README's HuggingFace-standard contract.
///
/// Env-var mutation is safe here: this assembly disables test parallelization (AssemblyInfo.cs).
/// </summary>
public sealed class CachePathResolutionTests
{
    [Fact]
    public void OnnxGeneratorModelFactory_Default_HonorsHfHubCacheEnv()
    {
        var original = Environment.GetEnvironmentVariable("HF_HUB_CACHE");
        var hub = Path.Combine(Path.GetTempPath(), "lmsupply-tests", "hf-hub-p2");

        try
        {
            Environment.SetEnvironmentVariable("HF_HUB_CACHE", hub);

            using var factory = new OnnxGeneratorModelFactory();
            var cachePath = factory.GetModelCachePath("org/model");

            cachePath.Should().StartWith(hub,
                "the arg-less factory must resolve its default cache via CacheManager's HF env chain");
        }
        finally
        {
            Environment.SetEnvironmentVariable("HF_HUB_CACHE", original);
        }
    }

    [Fact]
    public void OnnxGeneratorModelFactory_Default_MatchesCanonicalResolver()
    {
        using var factory = new OnnxGeneratorModelFactory();
        var cachePath = factory.GetModelCachePath("org/model");

        cachePath.Should().StartWith(CacheManager.GetDefaultCacheDirectory(),
            "whatever the environment, the factory default and the canonical resolver must agree");
    }
}
