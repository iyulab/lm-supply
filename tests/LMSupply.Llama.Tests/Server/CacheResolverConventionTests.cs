using System.Reflection;
using AwesomeAssertions;
using LMSupply.Llama.Server;
using Xunit;

namespace LMSupply.Llama.Tests.Server;

/// <summary>
/// Convention teeth for the single-resolver rule (Forge upstream-019): LMSupply.Llama's
/// binary store paths must resolve through LMSupplyCachePaths (LMSupply.Core) — the original
/// hardcoded defaults here were unreachable by any override and grew unboundedly. This
/// assembly must declare no <c>GetDefaultCacheDirectory</c> of its own.
/// </summary>
public sealed class CacheResolverConventionTests
{
    [Fact]
    public void Llama_HasNoPrivateDefaultCacheResolverCopies()
    {
        var offenders = typeof(LlamaServerDownloader).Assembly.GetTypes()
            .SelectMany(type => type.GetMethods(
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance |
                BindingFlags.DeclaredOnly))
            .Where(method => method.Name == "GetDefaultCacheDirectory")
            .Select(method => $"{method.DeclaringType!.FullName}.{method.Name}")
            .ToList();

        offenders.Should().BeEmpty(
            "default cache paths are resolved by LMSupply.Core (LMSupplyCachePaths for non-HF " +
            "artifacts) — private copies split the cache");
    }
}
