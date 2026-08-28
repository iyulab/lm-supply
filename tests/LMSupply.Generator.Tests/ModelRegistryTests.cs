using AwesomeAssertions;
using LMSupply.Exceptions;

namespace LMSupply.Generator.Tests;

public class ModelRegistryTests
{
    // ===== New GeneratorModelRegistry tests =====

    [Fact]
    public void Resolve_WithPhiMiniAlias_ReturnsPhi4Mini()
    {
        // Act
        var model = GeneratorModelRegistry.Default.Resolve("phi-4-mini");

        // Assert
        model.Should().NotBeNull();
        model.ModelId.Should().Be("microsoft/Phi-4-mini-instruct-onnx");
    }

    [Fact]
    public void Resolve_WithFastAlias_ReturnsPhi4Mini()
    {
        // Fast alias now points to Phi4Mini (smallest FC-capable ONNX model)
        // Act
        var model = GeneratorModelRegistry.Default.Resolve("fast");

        // Assert
        model.Should().NotBeNull();
        model.ModelId.Should().Be("microsoft/Phi-4-mini-instruct-onnx");
    }

    [Fact]
    public void Resolve_WithQualityAlias_ReturnsPhi4()
    {
        // Act
        var model = GeneratorModelRegistry.Default.Resolve("quality");

        // Assert
        model.Should().NotBeNull();
        model.ModelId.Should().Be("microsoft/phi-4-onnx");
    }

    [Fact]
    public void TryResolve_WithAutoAlias_ReturnsHardwareAdaptive()
    {
        // Act
        var result = GeneratorModelRegistry.Default.TryResolve("auto", out var model);

        // Assert
        result.Should().BeTrue();
        model.Should().NotBeNull();
        model!.AliasName.Should().Be("auto");
    }

    [Fact]
    public void TryResolve_WithFullModelId_ReturnsModel()
    {
        // Act
        var result = GeneratorModelRegistry.Default.TryResolve("microsoft/Phi-4-mini-instruct-onnx", out var model);

        // Assert
        result.Should().BeTrue();
        model.Should().NotBeNull();
        model!.DisplayName.Should().Be("Phi-4 Mini");
    }

    [Fact]
    public void TryResolve_WithShortName_ReturnsModel()
    {
        // Act
        var result = GeneratorModelRegistry.Default.TryResolve("Phi-4-mini-instruct-onnx", out var model);

        // Assert
        result.Should().BeTrue();
        model.Should().NotBeNull();
        model!.ModelId.Should().Be("microsoft/Phi-4-mini-instruct-onnx");
    }

    [Fact]
    public void Resolve_WithUnknownAlias_Throws()
    {
        // Act
        var act = () => GeneratorModelRegistry.Default.Resolve("unknown-alias");

        // Assert
        act.Should().Throw<ModelNotFoundException>()
            .WithMessage("*unknown-alias*not found*");
    }

    [Fact]
    public void TryResolve_WithUnknownHFRepo_CreatesFallback()
    {
        // Act
        var result = GeneratorModelRegistry.Default.TryResolve("my-org/my-model", out var model);

        // Assert
        result.Should().BeTrue();
        model.Should().NotBeNull();
        model!.ModelId.Should().Be("my-org/my-model");
    }

    [Fact]
    public void GetAvailableModels_ReturnsAllRegisteredModels()
    {
        // Act
        var models = GeneratorModelRegistry.Default.GetAvailableModels();

        // Assert
        models.Should().NotBeEmpty();
        models.Should().HaveCount(3);
    }

    [Fact]
    public void GetAliases_ContainsStandardAliases()
    {
        // Act
        var aliases = GeneratorModelRegistry.Default.GetAliases();

        // Assert
        var aliasNames = aliases.Select(a => a.Name).ToList();
        aliasNames.Should().Contain("auto");
        aliasNames.Should().Contain("phi-4-mini");
        aliasNames.Should().Contain("fast");
        aliasNames.Should().Contain("quality");
    }

    [Fact]
    public void GetAvailableModels_AllHaveLicenseInfo()
    {
        // Act
        var models = GeneratorModelRegistry.Default.GetAvailableModels();

        // Assert
        models.Should().AllSatisfy(m =>
        {
            m.LicenseName.Should().NotBeNullOrEmpty();
            m.ChatFormat.Should().NotBeNullOrEmpty();
        });
    }

    [Fact]
    public void MITModels_HaveNoRestrictions()
    {
        // Act
        var models = GeneratorModelRegistry.Default.GetAvailableModels()
            .Where(m => m.License == LicenseTier.MIT);

        // Assert
        models.Should().NotBeEmpty();
        models.Should().AllSatisfy(m =>
        {
            m.HasRestrictions.Should().BeFalse();
        });
    }

    [Fact]
    public void AllModels_AreMITLicensed()
    {
        // All remaining ONNX models are MIT-licensed Phi-4 series
        // Act
        var models = GeneratorModelRegistry.Default.GetAvailableModels();

        // Assert
        models.Should().AllSatisfy(m =>
        {
            m.License.Should().Be(LicenseTier.MIT);
            m.HasRestrictions.Should().BeFalse();
        });
    }

    [Fact]
    public void RegisterAlias_ShouldCreateNewMapping()
    {
        // Arrange
        var registry = new GeneratorModelRegistry(DefaultGeneratorModels.All);

        // Act
        registry.RegisterAlias("my-gen", "microsoft/Phi-4-mini-instruct-onnx");
        var result = registry.TryResolve("my-gen", out var model);

        // Assert
        result.Should().BeTrue();
        model!.ModelId.Should().Be("microsoft/Phi-4-mini-instruct-onnx");
    }

    [Fact]
    public void RegisterAlias_WithSystemAliasName_ShouldThrow()
    {
        // Arrange
        var registry = new GeneratorModelRegistry(DefaultGeneratorModels.All);

        // Act
        var act = () => registry.RegisterAlias("phi-4-mini", "microsoft/phi-4-onnx");

        // Assert
        act.Should().Throw<AliasConflictException>();
    }

    [Fact]
    public void DefaultGeneratorModels_All_ShouldContainAllModels()
    {
        // Assert
        DefaultGeneratorModels.All.Should().HaveCount(3);
        DefaultGeneratorModels.All.Should().Contain(DefaultGeneratorModels.Phi4Mini);
        DefaultGeneratorModels.All.Should().Contain(DefaultGeneratorModels.Phi35Mini);
        DefaultGeneratorModels.All.Should().Contain(DefaultGeneratorModels.Phi4);
    }

    [Fact]
    public void DefaultGeneratorModels_Default_ShouldBePhi4Mini()
    {
        // Assert
        DefaultGeneratorModels.Default.Should().Be(DefaultGeneratorModels.Phi4Mini);
    }

    [Fact]
    public void DefaultGeneratorModels_Fast_ShouldBePhi4Mini()
    {
        // Phi4Mini is the smallest FC-capable ONNX model
        DefaultGeneratorModels.Fast.Should().Be(DefaultGeneratorModels.Phi4Mini);
    }

    [Fact]
    public void LocalGenerator_Registry_ShouldExposeRegistry()
    {
        // Act
        var registry = LocalGenerator.Registry;

        // Assert
        registry.Should().NotBeNull();
        registry.Should().BeOfType<GeneratorModelRegistry>();
    }

    [Fact]
    public void ModelInfo_GetMemoryConfig_ReturnsValidConfig()
    {
        // Arrange
        var model = GeneratorModelRegistry.Default.Resolve("phi-4-mini");

        // Act
        var config = model.GetMemoryConfig();

        // Assert
        config.ParameterCount.Should().Be(model.ParameterCount);
        config.NumLayers.Should().Be(model.NumLayers);
        config.HiddenSize.Should().Be(model.HiddenSize);
        config.Quantization.Should().Be(model.DefaultQuantization);
    }

    [Fact]
    public void AllModels_HaveValidChatFormat()
    {
        // All remaining models use phi3 chat format
        // Act
        var models = GeneratorModelRegistry.Default.GetAvailableModels();

        // Assert
        models.Should().AllSatisfy(m =>
        {
            m.ChatFormat.Should().NotBeNullOrEmpty();
            m.ChatFormat.Should().Be("phi3");
        });
    }

    [Fact]
    public void GetAvailableModels_FilterByUnrestricted_ReturnsMITOnly()
    {
        var models = GeneratorModelRegistry.Default.GetAvailableModels()
            .Where(m => !m.HasRestrictions)
            .ToList();

        models.Should().NotBeEmpty();
        models.Should().AllSatisfy(m =>
        {
            m.License.Should().Be(LicenseTier.MIT);
            m.HasRestrictions.Should().BeFalse();
        });
    }

    [Theory]
    [InlineData("microsoft/Phi-4-mini-instruct-onnx", LicenseTier.MIT)]
    [InlineData("microsoft/Phi-3.5-mini-instruct-onnx", LicenseTier.MIT)]
    [InlineData("microsoft/phi-4-onnx", LicenseTier.MIT)]
    public void TryResolve_KnownModel_HasCorrectLicense(string modelId, LicenseTier expectedLicense)
    {
        var result = GeneratorModelRegistry.Default.TryResolve(modelId, out var model);
        result.Should().BeTrue();
        model!.License.Should().Be(expectedLicense);
    }

    [Fact]
    public void TryResolve_NonExistentNonRepoFormat_ReturnsFalse()
    {
        // Non-repo-format unknown string (no slash) should not resolve
        var result = GeneratorModelRegistry.Default.TryResolve("completely-unknown-model", out _);
        result.Should().BeFalse();
    }
}
