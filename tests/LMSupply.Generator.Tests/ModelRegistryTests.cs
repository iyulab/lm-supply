using FluentAssertions;

namespace LMSupply.Generator.Tests;

public class ModelRegistryTests
{
    [Fact]
    public void GetAllModels_ReturnsAllRegisteredModels()
    {
        // Act
        var models = ModelRegistry.GetAllModels();

        // Assert
        models.Should().NotBeEmpty();
        models.Should().HaveCountGreaterOrEqualTo(5);
    }

    [Fact]
    public void GetUnrestrictedModels_ReturnsMITLicensedOnly()
    {
        // Act
        var models = ModelRegistry.GetUnrestrictedModels();

        // Assert
        models.Should().NotBeEmpty();
        models.Should().AllSatisfy(m =>
        {
            m.License.Should().Be(LicenseTier.MIT);
            m.HasRestrictions.Should().BeFalse();
        });
    }

    [Fact]
    public void GetModelsByLicense_Conditional_ReturnsRestrictedModels()
    {
        // Act
        var models = ModelRegistry.GetModelsByLicense(LicenseTier.Conditional);

        // Assert
        models.Should().NotBeEmpty();
        models.Should().AllSatisfy(m =>
        {
            m.License.Should().Be(LicenseTier.Conditional);
            m.HasRestrictions.Should().BeTrue();
            m.LicenseRestrictions.Should().NotBeNullOrEmpty();
        });
    }

    [Theory]
    [InlineData("microsoft/Phi-3.5-mini-instruct-onnx", LicenseTier.MIT)]
    [InlineData("microsoft/phi-4-onnx", LicenseTier.MIT)]
    [InlineData("onnx-community/Llama-3.2-1B-Instruct-ONNX", LicenseTier.Conditional)]
    [InlineData("onnx-community/Llama-3.2-3B-Instruct-ONNX", LicenseTier.Conditional)]
    public void GetModel_ReturnsCorrectLicense(string modelId, LicenseTier expectedLicense)
    {
        // Act
        var model = ModelRegistry.GetModel(modelId);

        // Assert
        model.Should().NotBeNull();
        model!.License.Should().Be(expectedLicense);
    }

    [Fact]
    public void GetModel_UnknownModel_ReturnsNull()
    {
        // Act
        var model = ModelRegistry.GetModel("unknown/model");

        // Assert
        model.Should().BeNull();
    }

    [Fact]
    public void IsRegistered_KnownModel_ReturnsTrue()
    {
        // Act
        var isRegistered = ModelRegistry.IsRegistered("microsoft/Phi-3.5-mini-instruct-onnx");

        // Assert
        isRegistered.Should().BeTrue();
    }

    [Fact]
    public void IsRegistered_UnknownModel_ReturnsFalse()
    {
        // Act
        var isRegistered = ModelRegistry.IsRegistered("unknown/model");

        // Assert
        isRegistered.Should().BeFalse();
    }

    [Fact]
    public void GetModelsForMemory_8GB_ReturnsSmallModels()
    {
        // Arrange
        var availableMemory = 8L * 1024 * 1024 * 1024; // 8GB

        // Act
        var models = ModelRegistry.GetModelsForMemory(availableMemory);

        // Assert
        models.Should().NotBeEmpty();
        // Should include smaller models
        models.Should().Contain(m => m.ModelId.Contains("Llama-3.2-1B"));
    }

    [Fact]
    public void GetDefaultModel_ReturnsValidModel()
    {
        // Act
        var model = ModelRegistry.GetDefaultModel();

        // Assert
        model.Should().NotBeNull();
        model.ModelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public void ModelInfo_GetMemoryConfig_ReturnsValidConfig()
    {
        // Arrange
        var model = ModelRegistry.GetModel("microsoft/Phi-3.5-mini-instruct-onnx")!;

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
        // Act
        var models = ModelRegistry.GetAllModels();

        // Assert
        models.Should().AllSatisfy(m =>
        {
            m.ChatFormat.Should().NotBeNullOrEmpty();
            m.ChatFormat.Should().BeOneOf("phi3", "llama3", "gemma", "chatml");
        });
    }

    [Fact]
    public void Phi35Mini_IsMITLicensed()
    {
        // Act
        var model = ModelRegistry.GetModel("microsoft/Phi-3.5-mini-instruct-onnx");

        // Assert
        model.Should().NotBeNull();
        model!.License.Should().Be(LicenseTier.MIT);
        model.LicenseName.Should().Be("MIT");
        model.HasRestrictions.Should().BeFalse();
    }

    [Fact]
    public void LlamaModels_HaveMAURestriction()
    {
        // Act
        var model = ModelRegistry.GetModel("onnx-community/Llama-3.2-1B-Instruct-ONNX");

        // Assert
        model.Should().NotBeNull();
        model!.License.Should().Be(LicenseTier.Conditional);
        model.LicenseRestrictions.Should().Contain("MAU");
    }
}
