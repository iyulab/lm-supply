using FluentAssertions;
using LMSupply.ImageGenerator.Models;

namespace LMSupply.ImageGenerator.Tests;

public class WellKnownImageModelsTests
{
    [Theory]
    [InlineData("default")]
    [InlineData("fast")]
    [InlineData("quality")]
    [InlineData("lcm-dreamshaper-v7")]
    public void Resolve_WithKnownAlias_ReturnsModelDefinition(string alias)
    {
        // Act
        var result = WellKnownImageModels.Resolve(alias);

        // Assert
        result.RepoId.Should().NotBeNullOrEmpty();
        result.RecommendedSteps.Should().BeInRange(1, 10);
        result.RecommendedGuidanceScale.Should().BeInRange(0.5f, 5f);
    }

    [Theory]
    [InlineData("Default")]
    [InlineData("DEFAULT")]
    [InlineData("Fast")]
    [InlineData("FAST")]
    public void Resolve_WithCaseInsensitiveAlias_ReturnsModelDefinition(string alias)
    {
        // Act
        var result = WellKnownImageModels.Resolve(alias);

        // Assert
        result.RepoId.Should().NotBeNullOrEmpty();
    }

    [Theory]
    [InlineData("owner/model-name")]
    [InlineData("TheyCallMeHex/SomeOther-Model")]
    public void Resolve_WithUnknownRepoId_ReturnsWithDefaultSettings(string repoId)
    {
        // Act
        var result = WellKnownImageModels.Resolve(repoId);

        // Assert
        result.RepoId.Should().Be(repoId);
        result.RecommendedSteps.Should().Be(4);
        result.RecommendedGuidanceScale.Should().Be(1.0f);
    }

    [Fact]
    public void Resolve_WithNullOrEmpty_ThrowsException()
    {
        // Act & Assert
        var actNull = () => WellKnownImageModels.Resolve(null!);
        var actEmpty = () => WellKnownImageModels.Resolve(string.Empty);
        var actWhitespace = () => WellKnownImageModels.Resolve("   ");

        actNull.Should().Throw<ArgumentException>();
        actEmpty.Should().Throw<ArgumentException>();
        actWhitespace.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void GetAliases_ReturnsNonEmptyCollection()
    {
        // Act
        var aliases = WellKnownImageModels.GetAliases();

        // Assert
        aliases.Should().NotBeEmpty();
        aliases.Should().Contain("default");
        aliases.Should().Contain("fast");
        aliases.Should().Contain("quality");
    }

    [Theory]
    [InlineData("default", true)]
    [InlineData("fast", true)]
    [InlineData("quality", true)]
    [InlineData("unknown-model", false)]
    [InlineData("owner/repo", false)]
    public void IsAlias_ReturnsCorrectResult(string input, bool expected)
    {
        // Act
        var result = WellKnownImageModels.IsAlias(input);

        // Assert
        result.Should().Be(expected);
    }

    [Fact]
    public void DefaultAlias_HasOptimalSettings()
    {
        // Act
        var result = WellKnownImageModels.Resolve("default");

        // Assert
        result.RecommendedSteps.Should().Be(4);
        result.RecommendedGuidanceScale.Should().Be(1.0f);
    }

    [Fact]
    public void FastAlias_HasFewerSteps()
    {
        // Act
        var fast = WellKnownImageModels.Resolve("fast");
        var defaultModel = WellKnownImageModels.Resolve("default");

        // Assert
        fast.RecommendedSteps.Should().BeLessThanOrEqualTo(defaultModel.RecommendedSteps);
    }
}
