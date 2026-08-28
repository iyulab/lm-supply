using AwesomeAssertions;
using Xunit;

namespace LMSupply.Generator.Tests;

/// <summary>
/// Verifies the Generator registry's Default factory applies the user alias configuration
/// file (issue: alias-config-wiring AC#1). Calls the production factory
/// (<see cref="GeneratorModelRegistry.CreateDefault"/>) with a hermetic config file via
/// LMSUPPLY_ALIASES_FILE — the static Default singleton itself cannot be re-initialized
/// per test, so the factory is the testable seam.
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
    public void CreateDefault_AppliesGeneratorDomainFromConfigFile()
    {
        File.WriteAllText(Path.Combine(_dir, "aliases.json"), """
            { "generator": { "my-writer": "org/custom-writer" } }
            """);

        var registry = GeneratorModelRegistry.CreateDefault();

        registry.TryResolve("my-writer", out var resolved).Should().BeTrue(
            "a user alias defined in aliases.json must resolve with zero consumer code");
        resolved!.ModelId.Should().Be("org/custom-writer");
    }

    [Fact]
    public void RegisterAlias_AfterFileLoad_OverridesFileEntry()
    {
        File.WriteAllText(Path.Combine(_dir, "aliases.json"), """
            { "generator": { "my-writer": "org/from-file" } }
            """);

        var registry = GeneratorModelRegistry.CreateDefault();
        registry.RegisterAlias("my-writer", "org/from-code");

        registry.TryResolve("my-writer", out var resolved).Should().BeTrue();
        resolved!.ModelId.Should().Be("org/from-code",
            "programmatic registration must win over the config file (AC#4)");
    }
}
