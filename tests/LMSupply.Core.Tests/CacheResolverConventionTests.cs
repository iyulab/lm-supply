using System.Reflection;
using AwesomeAssertions;
using LMSupply.Download;

namespace LMSupply.Core.Tests;

/// <summary>
/// Convention teeth for the single-resolver rule (Forge upstream-019): default cache
/// directories are decided in exactly two places — <see cref="CacheManager"/> (HF model hub,
/// HF env chain) and <see cref="LMSupplyCachePaths"/> (non-HF artifact root, LMSUPPLY_CACHE_DIR).
/// Any other type declaring its own <c>GetDefaultCacheDirectory</c> is a private copy — the
/// pattern that split the cache across four disagreeing implementations before unification.
/// </summary>
public sealed class CacheResolverConventionTests
{
    [Fact]
    public void Core_HasNoPrivateDefaultCacheResolverCopies()
    {
        var offenders = FindResolverCopies(typeof(CacheManager).Assembly,
            allowed: [typeof(CacheManager), typeof(LMSupplyCachePaths)]);

        offenders.Should().BeEmpty(
            "every default cache path must resolve through CacheManager (HF models) or " +
            "LMSupplyCachePaths (non-HF artifacts) — private copies split the cache");
    }

    internal static IReadOnlyList<string> FindResolverCopies(Assembly assembly, Type[] allowed)
        => assembly.GetTypes()
            .Where(type => !allowed.Contains(type))
            .SelectMany(type => type.GetMethods(
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance |
                BindingFlags.DeclaredOnly))
            .Where(method => method.Name == "GetDefaultCacheDirectory")
            .Select(method => $"{method.DeclaringType!.FullName}.{method.Name}")
            .ToList();
}
