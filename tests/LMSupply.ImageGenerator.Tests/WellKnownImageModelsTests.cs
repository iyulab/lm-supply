using AwesomeAssertions;
using LMSupply.Exceptions;
using LMSupply.ImageGenerator.Models;

namespace LMSupply.ImageGenerator.Tests;

public class WellKnownImageModelsTests
{
    // ===== New registry tests =====

    [Theory]
    [InlineData("default")]
    [InlineData("fast")]
    [InlineData("quality")]
    [InlineData("lcm-dreamshaper-v7")]
    [InlineData("lcm-ssd-1b")]
    public void Resolve_WithKnownAlias_ReturnsModelInfo(string alias)
    {
        // Act
        var result = ImageGeneratorModelRegistry.Default.Resolve(alias);

        // Assert
        result.Should().NotBeNull();
        result.ModelId.Should().NotBeNullOrEmpty();
        result.RecommendedSteps.Should().BeInRange(1, 10);
        result.RecommendedGuidanceScale.Should().BeInRange(0.5f, 5f);
    }

    [Theory]
    [InlineData("Default")]
    [InlineData("DEFAULT")]
    [InlineData("Fast")]
    [InlineData("FAST")]
    public void Resolve_CaseInsensitive_ReturnsModelInfo(string alias)
    {
        // Act
        var result = ImageGeneratorModelRegistry.Default.Resolve(alias);

        // Assert
        result.Should().NotBeNull();
        result.ModelId.Should().NotBeNullOrEmpty();
    }

    [Theory]
    [InlineData("owner/model-name")]
    [InlineData("TheyCallMeHex/SomeOther-Model")]
    public void Resolve_WithUnknownRepoId_CreatesFallback(string repoId)
    {
        // Act
        var result = ImageGeneratorModelRegistry.Default.Resolve(repoId);

        // Assert
        result.ModelId.Should().Be(repoId);
        result.RecommendedSteps.Should().Be(4);
        result.RecommendedGuidanceScale.Should().Be(1.0f);
    }

    [Fact]
    public void Resolve_WithUnknownAlias_ThrowsModelNotFoundException()
    {
        // Act & Assert
        var act = () => ImageGeneratorModelRegistry.Default.Resolve("unknown-alias");

        act.Should().Throw<ModelNotFoundException>()
            .WithMessage("*unknown-alias*not found*");
    }

    [Fact]
    public void TryResolve_WithAutoAlias_ReturnsHardwareAdaptive()
    {
        // Act
        var result = ImageGeneratorModelRegistry.Default.TryResolve("auto", out var modelInfo);

        // Assert
        result.Should().BeTrue();
        modelInfo.Should().NotBeNull();
        modelInfo!.AliasName.Should().Be("auto");
    }

    [Fact]
    public void GetAliases_ContainsStandardAliases()
    {
        // Act
        var aliases = ImageGeneratorModelRegistry.Default.GetAliases();

        // Assert
        var aliasNames = aliases.Select(a => a.Name).ToList();
        aliasNames.Should().Contain("auto");
        aliasNames.Should().Contain("default");
        aliasNames.Should().Contain("fast");
        aliasNames.Should().Contain("quality");
        aliasNames.Should().Contain("lcm-dreamshaper-v7");
        aliasNames.Should().Contain("lcm-ssd-1b");
    }

    [Fact]
    public void GetAvailableModels_ReturnsRegisteredModels()
    {
        // Act
        var models = ImageGeneratorModelRegistry.Default.GetAvailableModels();

        // Assert
        models.Should().NotBeEmpty();
        // 5 entries total, but some share the same ModelId (only unique IDs)
        models.Select(m => m.ModelId).Distinct().Should().HaveCountGreaterThanOrEqualTo(2);
    }

    [Fact]
    public void DefaultAlias_HasOptimalSettings()
    {
        // Act
        var result = ImageGeneratorModelRegistry.Default.Resolve("default");

        // Assert
        result.RecommendedSteps.Should().Be(4);
        result.RecommendedGuidanceScale.Should().Be(1.0f);
    }

    [Fact]
    public void FastAlias_HasFewerSteps()
    {
        // Act
        var fast = ImageGeneratorModelRegistry.Default.Resolve("fast");
        var defaultModel = ImageGeneratorModelRegistry.Default.Resolve("default");

        // Assert
        fast.RecommendedSteps.Should().BeLessThanOrEqualTo(defaultModel.RecommendedSteps);
    }

    [Fact]
    public void RegisterAlias_ShouldCreateNewMapping()
    {
        // Arrange
        var registry = new ImageGeneratorModelRegistry(DefaultModels.All);

        // Act
        registry.RegisterAlias("my-gen", "TheyCallMeHex/LCM-Dreamshaper-V7-ONNX");
        var result = registry.TryResolve("my-gen", out var model);

        // Assert
        result.Should().BeTrue();
        model!.ModelId.Should().Be("TheyCallMeHex/LCM-Dreamshaper-V7-ONNX");
    }

    [Fact]
    public void DefaultModels_All_ShouldContainAllModels()
    {
        // Assert
        DefaultModels.All.Should().HaveCount(5);
        DefaultModels.All.Should().Contain(DefaultModels.LcmDreamshaperV7);
        DefaultModels.All.Should().Contain(DefaultModels.LcmDreamshaperV7Fast);
        DefaultModels.All.Should().Contain(DefaultModels.LcmDreamshaperV7Quality);
        DefaultModels.All.Should().Contain(DefaultModels.LcmDreamshaperV7Direct);
        DefaultModels.All.Should().Contain(DefaultModels.LcmSsd1B);
    }

    [Fact]
    public void DefaultModels_Default_ShouldBeLcmDreamshaperV7()
    {
        // Assert
        DefaultModels.Default.Should().Be(DefaultModels.LcmDreamshaperV7);
    }

    [Fact]
    public void LocalImageGenerator_Registry_ShouldExposeRegistry()
    {
        // Act
        var registry = LocalImageGenerator.Registry;

        // Assert
        registry.Should().NotBeNull();
        registry.Should().BeOfType<ImageGeneratorModelRegistry>();
    }

    [Fact]
    public void ToModelDefinition_ConvertsCorrectly()
    {
        // Arrange
        var info = DefaultModels.LcmDreamshaperV7;

        // Act
        var def = ImageGeneratorModelRegistry.ToModelDefinition(info);

        // Assert
        def.RepoId.Should().Be(info.ModelId);
        def.FriendlyName.Should().Be(info.ModelName);
        def.RecommendedSteps.Should().Be(info.RecommendedSteps);
        def.RecommendedGuidanceScale.Should().Be(info.RecommendedGuidanceScale);
    }

    [Theory]
    [InlineData("default", true)]
    [InlineData("fast", true)]
    [InlineData("quality", true)]
    [InlineData("unknown-model", false)]
    [InlineData("owner/repo", false)]
    public void GetAliases_ContainsAlias_ReturnsExpected(string input, bool expected)
    {
        var isAlias = ImageGeneratorModelRegistry.Default
            .GetAliases()
            .Any(a => string.Equals(a.Name, input, StringComparison.OrdinalIgnoreCase));

        isAlias.Should().Be(expected);
    }
}
