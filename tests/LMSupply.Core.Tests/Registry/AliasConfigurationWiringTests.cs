using AwesomeAssertions;

namespace LMSupply.Core.Tests.Registry;

/// <summary>
/// Tests for the wiring half of user alias configuration (issue: alias-config-wiring).
/// <see cref="AliasConfiguration"/> shipped as an inert utility — parse + default path existed
/// but nothing loaded the file. These tests cover the completion surface:
/// env-overridable config path, the fail-soft <c>ApplyDomain</c> fluent helper that module
/// registries call at Default initialization, and the canonical domain keys.
///
/// Fail-soft contract (operator lens): a typo in a user's aliases.json must never crash app
/// startup — bad entries are skipped with a Trace warning, good entries still apply.
///
/// Shares the "lmsupply-cache-env" collection: mutates the process-global
/// LMSUPPLY_ALIASES_FILE env var.
/// </summary>
[Collection("lmsupply-cache-env")]
public sealed class AliasConfigurationWiringTests : IDisposable
{
    private readonly string _dir;
    private readonly string _file;
    private readonly string? _originalEnv;

    public AliasConfigurationWiringTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "lmsupply-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
        _file = Path.Combine(_dir, "aliases.json");
        _originalEnv = Environment.GetEnvironmentVariable(AliasConfiguration.FileEnvironmentVariable);
        Environment.SetEnvironmentVariable(AliasConfiguration.FileEnvironmentVariable, _file);
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(AliasConfiguration.FileEnvironmentVariable, _originalEnv);
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    private sealed class TestRegistry() : ModelRegistryBase<TestModelInfo>(
        [new TestModelInfo("org/sys-model", "sys")])
    {
        protected override TestModelInfo GetAutoModel() => new("org/sys-model", "sys");
        protected override TestModelInfo CreateFallbackModelInfo(string modelId) => new(modelId, modelId);
    }

    private sealed record TestModelInfo(string Id, string AliasName) : IModelInfoBase
    {
        public string? Description => null;
    }

    [Fact]
    public void DefaultFilePath_HonorsEnvOverride()
    {
        AliasConfiguration.DefaultFilePath.Should().Be(_file,
            "LMSUPPLY_ALIASES_FILE must relocate the config file (also what makes these tests hermetic)");
    }

    [Fact]
    public void ApplyDomain_AppliesOnlyTheRequestedDomain_AndReturnsSameInstance()
    {
        File.WriteAllText(_file, """
            {
              "generator": { "my-writer": "org/writer-model" },
              "embedder":  { "my-embed": "org/embed-model" }
            }
            """);

        var registry = new TestRegistry();
        var returned = AliasConfiguration.ApplyDomain(registry, "generator");

        returned.Should().BeSameAs(registry, "the helper is fluent for one-line Default wiring");
        registry.TryResolve("my-writer", out var resolved).Should().BeTrue();
        resolved!.Id.Should().Be("org/writer-model");
        registry.TryResolve("my-embed", out _).Should().BeFalse(
            "the embedder domain must not leak into a generator registry");
    }

    [Fact]
    public void ApplyDomain_MissingFileOrDomain_IsANoOp()
    {
        var registry = new TestRegistry();

        AliasConfiguration.ApplyDomain(registry, "generator"); // no file at all
        File.WriteAllText(_file, """{ "embedder": { "x": "org/y" } }""");
        AliasConfiguration.ApplyDomain(registry, "generator"); // file without the domain

        registry.GetAliases().Should().NotContain(a => a.Kind == AliasKind.User);
    }

    [Fact]
    public void ApplyDomain_BrokenJson_DoesNotThrow()
    {
        File.WriteAllText(_file, "{ not valid json !!");

        var registry = new TestRegistry();
        var act = () => AliasConfiguration.ApplyDomain(registry, "generator");

        act.Should().NotThrow("a config typo must never crash app startup (fail-soft)");
    }

    [Fact]
    public void ApplyDomain_SkipsBadEntries_ButAppliesGoodOnes()
    {
        File.WriteAllText(_file, """
            {
              "generator": {
                "sys": "org/hijack",
                "auto": "org/hijack",
                "bad:name": "org/whatever",
                "good": "org/good-model"
              }
            }
            """);

        var registry = new TestRegistry();
        var act = () => AliasConfiguration.ApplyDomain(registry, "generator");

        act.Should().NotThrow("conflicting entries are skipped, not fatal");
        registry.TryResolve("good", out var resolved).Should().BeTrue(
            "entries after a bad one must still be applied");
        resolved!.Id.Should().Be("org/good-model");
        registry.Resolve("sys").Id.Should().Be("org/sys-model",
            "a user entry must never hijack a system alias");
        registry.GetAliases().Should().NotContain(a => a.Name == "bad:name",
            "alias names containing ':' collide with the qualifier syntax and are skipped");
    }

    [Fact]
    public void Domains_CanonicalKeys_AreDefined()
    {
        AliasConfiguration.Domains.Generator.Should().Be("generator");
        AliasConfiguration.Domains.Embedder.Should().Be("embedder");
        AliasConfiguration.Domains.Reranker.Should().Be("reranker");
    }
}
