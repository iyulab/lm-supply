using AwesomeAssertions;

namespace LMSupply.Core.Tests.Registry;

/// <summary>
/// The user-alias resolution invariant (issue: alias-config-wiring AC#3 groundwork):
/// a registered user alias ALWAYS resolves. When its target is not otherwise resolvable
/// (e.g. a "gguf:*" domain identifier — no '/', not a path), the registry trusts the
/// user's explicit mapping and produces a fallback model info for the target, instead of
/// failing and letting downstream treat the alias name itself as a model id.
/// Also covers TryGetUserAliasTarget — the pre-translation seam Local* entry points use
/// so format detection (gguf prefix etc.) runs against the target, not the alias name.
/// </summary>
public class UserAliasResolutionInvariantTests
{
    private static TestModelRegistry CreateRegistry() => new(
    [
        new TestModelInfo { Id = "org/model-a", AliasName = "default", Description = null },
    ]);

    [Fact]
    public void UserAlias_WithUnresolvableTarget_ResolvesToFallbackOfTarget()
    {
        var registry = CreateRegistry();
        registry.RegisterAlias("my-writer", "gguf:qwen3-quality");

        registry.TryResolve("my-writer", out var resolved).Should().BeTrue(
            "a registered user alias must always resolve — the user's mapping is explicit");
        resolved!.Id.Should().Be("gguf:qwen3-quality",
            "the fallback must carry the TARGET, never the alias name");
    }

    [Fact]
    public void TryGetUserAliasTarget_ReturnsTargetForUserAlias()
    {
        var registry = CreateRegistry();
        registry.RegisterAlias("my-writer", "gguf:qwen3-quality");

        registry.TryGetUserAliasTarget("my-writer", out var target).Should().BeTrue();
        target.Should().Be("gguf:qwen3-quality");
    }

    [Fact]
    public void TryGetUserAliasTarget_FalseForSystemAliasAndUnknown()
    {
        var registry = CreateRegistry();

        registry.TryGetUserAliasTarget("default", out _).Should().BeFalse(
            "system aliases are not user aliases — callers must not pre-translate them");
        registry.TryGetUserAliasTarget("nope", out _).Should().BeFalse();
    }
}
