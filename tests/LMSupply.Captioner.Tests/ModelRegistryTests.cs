using FluentAssertions;
using LMSupply.Captioner.Models;
using LMSupply.Vision;

namespace LMSupply.Captioner.Tests;

public class ModelRegistryTests
{
    [Fact]
    public void TryGetModel_WithDefaultAlias_ShouldReturnVitGpt2()
    {
        // Act
        var result = ModelRegistry.TryGetModel("default", out var modelInfo);

        // Assert
        result.Should().BeTrue();
        modelInfo.Should().NotBeNull();
        modelInfo!.Alias.Should().Be("vit-gpt2");
    }

    [Fact]
    public void TryGetModel_WithVitGpt2Alias_ShouldReturnModel()
    {
        // Act
        var result = ModelRegistry.TryGetModel("vit-gpt2", out var modelInfo);

        // Assert
        result.Should().BeTrue();
        modelInfo.Should().NotBeNull();
        modelInfo!.RepoId.Should().Be("Xenova/vit-gpt2-image-captioning");
    }

    [Fact]
    public void TryGetModel_WithRepoId_ShouldReturnModel()
    {
        // Act
        var result = ModelRegistry.TryGetModel("Xenova/vit-gpt2-image-captioning", out var modelInfo);

        // Assert
        result.Should().BeTrue();
        modelInfo.Should().NotBeNull();
        modelInfo!.Alias.Should().Be("vit-gpt2");
    }

    [Fact]
    public void TryGetModel_CaseInsensitive_ShouldWork()
    {
        // Act
        var result1 = ModelRegistry.TryGetModel("VIT-GPT2", out var model1);
        var result2 = ModelRegistry.TryGetModel("Vit-Gpt2", out var model2);
        var result3 = ModelRegistry.TryGetModel("vit-gpt2", out var model3);

        // Assert
        result1.Should().BeTrue();
        result2.Should().BeTrue();
        result3.Should().BeTrue();
        model1.Should().Be(model2);
        model2.Should().Be(model3);
    }

    [Fact]
    public void TryGetModel_WithUnknownModel_ShouldReturnFalse()
    {
        // Act
        var result = ModelRegistry.TryGetModel("unknown-model", out var modelInfo);

        // Assert
        result.Should().BeFalse();
        modelInfo.Should().BeNull();
    }

    [Fact]
    public void GetModel_WithValidAlias_ShouldReturnModel()
    {
        // Act
        var model = ModelRegistry.GetModel("vit-gpt2");

        // Assert
        model.Should().NotBeNull();
        model.Alias.Should().Be("vit-gpt2");
    }

    [Fact]
    public void GetModel_WithUnknownModel_ShouldThrow()
    {
        // Act
        var act = () => ModelRegistry.GetModel("unknown-model");

        // Assert
        act.Should().Throw<KeyNotFoundException>()
            .WithMessage("*unknown-model*not found*");
    }

    [Fact]
    public void GetAvailableModels_ShouldReturnRegisteredAliases()
    {
        // Act
        var models = ModelRegistry.GetAvailableModels().ToList();

        // Assert
        models.Should().Contain("vit-gpt2");
    }

    [Fact]
    public void GetAvailableModels_ShouldReturnDistinctAliases()
    {
        // Act
        var models = ModelRegistry.GetAvailableModels().ToList();

        // Assert
        models.Should().OnlyHaveUniqueItems();
    }

    [Fact]
    public void GetAllModels_ShouldReturnModelInfoObjects()
    {
        // Act
        var models = ModelRegistry.GetAllModels().ToList();

        // Assert
        models.Should().NotBeEmpty();
        models.Should().AllSatisfy(m => m.Should().NotBeNull());
    }

    [Fact]
    public void VitGpt2Model_ShouldHaveCorrectConfiguration()
    {
        // Act
        var model = ModelRegistry.GetModel("vit-gpt2");

        // Assert
        model.EncoderFile.Should().Be("encoder_model.onnx");
        model.DecoderFile.Should().Be("decoder_model_merged.onnx");
        model.TokenizerType.Should().Be(TokenizerType.Gpt2);
        model.SupportsVqa.Should().BeFalse();
        model.VocabSize.Should().Be(50257);
        model.BosTokenId.Should().Be(50256);
        model.EosTokenId.Should().Be(50256);
        model.PadTokenId.Should().Be(50256);
    }

    [Fact]
    public void VitGpt2Model_ShouldUseViTGpt2PreprocessProfile()
    {
        // Act
        var model = ModelRegistry.GetModel("vit-gpt2");

        // Assert
        model.PreprocessProfile.Width.Should().Be(224);
        model.PreprocessProfile.Height.Should().Be(224);
    }

    [Fact]
    public void RegisterModel_ShouldAddNewModel()
    {
        // Arrange
        var newModel = new ModelInfo(
            RepoId: "test/test-model",
            Alias: "test-model",
            DisplayName: "Test Model",
            EncoderFile: "encoder.onnx",
            DecoderFile: "decoder.onnx",
            TokenizerType: TokenizerType.Gpt2,
            PreprocessProfile: PreprocessProfile.ImageNet,
            SupportsVqa: true,
            VocabSize: 10000,
            BosTokenId: 1,
            EosTokenId: 2,
            PadTokenId: 0);

        // Act
        ModelRegistry.RegisterModel(newModel);
        var result = ModelRegistry.TryGetModel("test-model", out var retrieved);

        // Assert
        result.Should().BeTrue();
        retrieved.Should().Be(newModel);
    }

    [Fact]
    public void RegisterModel_WithNullModel_ShouldThrow()
    {
        // Act
        var act = () => ModelRegistry.RegisterModel(null!);

        // Assert
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void RegisterAlias_ShouldCreateNewMapping()
    {
        // Arrange - ensure base model exists
        ModelRegistry.TryGetModel("vit-gpt2", out _).Should().BeTrue();

        // Act
        ModelRegistry.RegisterAlias("my-custom-alias", "vit-gpt2");
        var result = ModelRegistry.TryGetModel("my-custom-alias", out var model);

        // Assert
        result.Should().BeTrue();
        model!.Alias.Should().Be("vit-gpt2");
    }

    [Fact]
    public void RegisterAlias_WithNonexistentModel_ShouldThrow()
    {
        // Act
        var act = () => ModelRegistry.RegisterAlias("new-alias", "nonexistent-model");

        // Assert
        act.Should().Throw<ArgumentException>()
            .WithMessage("*nonexistent-model*not found*");
    }
}
