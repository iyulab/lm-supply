using AwesomeAssertions;
using LMSupply.Exceptions;

namespace LMSupply.Core.Tests.Registry;

// Test doubles (non-file-local so they can be used in public method signatures)
internal sealed record TestModelInfo : IModelInfoBase
{
    public required string Id { get; init; }
    public required string AliasName { get; init; }
    public string? Description { get; init; }
}

internal sealed class TestModelRegistry : ModelRegistryBase<TestModelInfo>
{
    public TestModelRegistry(IEnumerable<TestModelInfo> systemModels)
        : base(systemModels) { }

    protected override TestModelInfo GetAutoModel()
        => new() { Id = "auto/model", AliasName = "auto", Description = "Auto-selected" };

    protected override TestModelInfo CreateFallbackModelInfo(string modelId)
        => new() { Id = modelId, AliasName = modelId, Description = null };
}

public class ModelRegistryBaseTests
{
    private static TestModelRegistry CreateRegistry()
    {
        var models = new[]
        {
            new TestModelInfo { Id = "org/model-a", AliasName = "default", Description = "Default model" },
            new TestModelInfo { Id = "org/model-b", AliasName = "fast", Description = "Fast model" },
            new TestModelInfo { Id = "org/model-c", AliasName = "quality", Description = "Quality model" },
        };
        return new TestModelRegistry(models);
    }

    // --- System alias resolution ---

    [Theory]
    [InlineData("default", "org/model-a")]
    [InlineData("fast", "org/model-b")]
    [InlineData("quality", "org/model-c")]
    public void Resolve_SystemAlias_ReturnsCorrectModel(string alias, string expectedId)
    {
        var registry = CreateRegistry();
        var result = registry.Resolve(alias);
        result.Id.Should().Be(expectedId);
    }

    [Fact]
    public void Resolve_Auto_ReturnsAutoModel()
    {
        var registry = CreateRegistry();
        var result = registry.Resolve("auto");
        result.Id.Should().Be("auto/model");
    }

    [Theory]
    [InlineData("DEFAULT")]
    [InlineData("Default")]
    [InlineData("FAST")]
    public void Resolve_CaseInsensitive(string alias)
    {
        var registry = CreateRegistry();
        var result = registry.Resolve(alias);
        result.Should().NotBeNull();
    }

    // --- Full ID and short name resolution ---

    [Fact]
    public void Resolve_FullModelId_ReturnsModel()
    {
        var registry = CreateRegistry();
        var result = registry.Resolve("org/model-a");
        result.Id.Should().Be("org/model-a");
    }

    [Fact]
    public void Resolve_ShortName_ReturnsModel()
    {
        var registry = CreateRegistry();
        var result = registry.Resolve("model-b");
        result.Id.Should().Be("org/model-b");
    }

    // --- HuggingFace repo fallback ---

    [Fact]
    public void Resolve_UnknownHFRepo_CreatesFallback()
    {
        var registry = CreateRegistry();
        var result = registry.Resolve("unknown-org/unknown-model");
        result.Id.Should().Be("unknown-org/unknown-model");
    }

    // --- Not found ---

    [Fact]
    public void Resolve_UnknownNonRepoString_ThrowsModelNotFound()
    {
        var registry = CreateRegistry();
        var act = () => registry.Resolve("nonexistent");
        act.Should().Throw<ModelNotFoundException>();
    }

    // --- TryResolve ---

    [Fact]
    public void TryResolve_KnownAlias_ReturnsTrue()
    {
        var registry = CreateRegistry();
        registry.TryResolve("default", out var info).Should().BeTrue();
        info!.Id.Should().Be("org/model-a");
    }

    [Fact]
    public void TryResolve_Unknown_ReturnsFalse()
    {
        var registry = CreateRegistry();
        registry.TryResolve("nonexistent", out var info).Should().BeFalse();
        info.Should().BeNull();
    }

    // --- User alias registration ---

    [Fact]
    public void RegisterAlias_NewAlias_ResolvesCorrectly()
    {
        var registry = CreateRegistry();
        registry.RegisterAlias("my-model", "org/model-b");
        var result = registry.Resolve("my-model");
        result.Id.Should().Be("org/model-b");
    }

    [Fact]
    public void RegisterAlias_TargetingSystemAlias_ResolvesCorrectly()
    {
        var registry = CreateRegistry();
        registry.RegisterAlias("my-default", "default");
        var result = registry.Resolve("my-default");
        result.Id.Should().Be("org/model-a");
    }

    [Fact]
    public void RegisterAlias_ConflictsWithSystemAlias_Throws()
    {
        var registry = CreateRegistry();
        var act = () => registry.RegisterAlias("default", "org/model-b");
        act.Should().Throw<AliasConflictException>()
            .Which.AliasName.Should().Be("default");
    }

    [Fact]
    public void RegisterAlias_ConflictsWithAuto_Throws()
    {
        var registry = CreateRegistry();
        var act = () => registry.RegisterAlias("auto", "org/model-b");
        act.Should().Throw<AliasConflictException>();
    }

    [Fact]
    public void RegisterAlias_TargetingUserAlias_ThrowsChainException()
    {
        var registry = CreateRegistry();
        registry.RegisterAlias("alias-a", "org/model-a");
        var act = () => registry.RegisterAlias("alias-b", "alias-a");
        act.Should().Throw<AliasChainException>()
            .Which.TargetAlias.Should().Be("alias-a");
    }

    // --- RemoveAlias ---

    [Fact]
    public void RemoveAlias_ExistingUserAlias_ReturnsTrue()
    {
        var registry = CreateRegistry();
        registry.RegisterAlias("my-model", "org/model-a");
        registry.RemoveAlias("my-model").Should().BeTrue();
        registry.TryResolve("my-model", out _).Should().BeFalse();
    }

    [Fact]
    public void RemoveAlias_NonExistent_ReturnsFalse()
    {
        var registry = CreateRegistry();
        registry.RemoveAlias("nonexistent").Should().BeFalse();
    }

    [Fact]
    public void RemoveAlias_SystemAlias_ReturnsFalse()
    {
        var registry = CreateRegistry();
        registry.RemoveAlias("default").Should().BeFalse();
    }

    // --- GetAliases ---

    [Fact]
    public void GetAliases_ReturnsSystemAndUserAliases()
    {
        var registry = CreateRegistry();
        registry.RegisterAlias("my-model", "org/model-b");

        var aliases = registry.GetAliases();
        aliases.Should().Contain(a => a.Name == "default" && a.Kind == AliasKind.System);
        aliases.Should().Contain(a => a.Name == "fast" && a.Kind == AliasKind.System);
        aliases.Should().Contain(a => a.Name == "quality" && a.Kind == AliasKind.System);
        aliases.Should().Contain(a => a.Name == "auto" && a.Kind == AliasKind.System);
        aliases.Should().Contain(a => a.Name == "my-model" && a.Kind == AliasKind.User);
    }

    // --- GetAvailableModels ---

    [Fact]
    public void GetAvailableModels_ReturnsAllSystemModels()
    {
        var registry = CreateRegistry();
        var models = registry.GetAvailableModels();
        models.Should().HaveCount(3);
        models.Select(m => m.Id).Should().Contain("org/model-a");
    }

    // --- Edge cases ---

    [Fact]
    public void Resolve_NullOrEmpty_ThrowsOrReturnsFalse()
    {
        var registry = CreateRegistry();
        registry.TryResolve("", out _).Should().BeFalse();
        registry.TryResolve(null!, out _).Should().BeFalse();
    }

    [Fact]
    public void RegisterAlias_OverwriteExistingUserAlias_Updates()
    {
        var registry = CreateRegistry();
        registry.RegisterAlias("my-model", "org/model-a");
        registry.RegisterAlias("my-model", "org/model-b");
        registry.Resolve("my-model").Id.Should().Be("org/model-b");
    }
}
