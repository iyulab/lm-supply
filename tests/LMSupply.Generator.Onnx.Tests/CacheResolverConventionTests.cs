using System.Reflection;
using AwesomeAssertions;
using Xunit;

namespace LMSupply.Generator.Onnx.Tests;

/// <summary>
/// Convention teeth for the single-resolver rule (Forge upstream-019): LMSupply.Generator
/// carried two private <c>GetDefaultCacheDirectory</c> copies that hardcoded the hub path
/// without the HF env chain, splitting the model cache for HF-env users. Resolution belongs
/// to LMSupply.Core (CacheManager / LMSupplyCachePaths) — this assembly must declare none.
/// </summary>
public sealed class CacheResolverConventionTests
{
    [Fact]
    public void Generator_HasNoPrivateDefaultCacheResolverCopies()
    {
        var offenders = typeof(OnnxGeneratorModelFactory).Assembly.GetTypes()
            .SelectMany(type => type.GetMethods(
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance |
                BindingFlags.DeclaredOnly))
            .Where(method => method.Name == "GetDefaultCacheDirectory")
            .Select(method => $"{method.DeclaringType!.FullName}.{method.Name}")
            .ToList();

        offenders.Should().BeEmpty(
            "default cache paths are resolved by LMSupply.Core (CacheManager for HF models, " +
            "LMSupplyCachePaths for non-HF artifacts) — private copies split the cache");
    }
}
