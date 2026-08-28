using AwesomeAssertions;
using LMSupply.Embedder.Utils;
using Xunit;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// Verifies the Embedder registry's Default factory applies the user alias configuration
/// file (issue: alias-config-wiring AC#1) — mirror of the Generator-side wiring test.
/// Only this class reads LMSUPPLY_ALIASES_FILE in this assembly, so parallel classes
/// cannot observe the env mutation.
/// </summary>
public sealed class UserAliasWiringTests : IDisposable
{
    private readonly string _dir;
    private readonly string? _originalEnv;

    public UserAliasWiringTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "lmsupply-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
        _originalEnv = Environment.GetEnvironmentVariable(AliasConfiguration.FileEnvironmentVariable);
        Environment.SetEnvironmentVariable(
            AliasConfiguration.FileEnvironmentVariable, Path.Combine(_dir, "aliases.json"));
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(AliasConfiguration.FileEnvironmentVariable, _originalEnv);
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    [Fact]
    public void CreateDefault_AppliesEmbedderDomainFromConfigFile()
    {
        File.WriteAllText(Path.Combine(_dir, "aliases.json"), """
            { "embedder": { "my-embed": "org/custom-embedder" } }
            """);

        var registry = EmbedderModelRegistry.CreateDefault();

        registry.TryResolve("my-embed", out var resolved).Should().BeTrue(
            "a user alias defined in aliases.json must resolve with zero consumer code");
        resolved!.RepoId.Should().Be("org/custom-embedder");
    }
}
