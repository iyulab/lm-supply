using FluentAssertions;
using LMSupply.Ocr.Models;

namespace LMSupply.Ocr.Tests;

public class ModelRegistryTests
{
    #region Detection Model Tests

    [Fact]
    public void TryGetDetectionModel_WithDefaultAlias_ShouldReturnDbNetV3()
    {
        // Act
        var result = ModelRegistry.TryGetDetectionModel("default", out var modelInfo);

        // Assert
        result.Should().BeTrue();
        modelInfo.Should().NotBeNull();
        modelInfo!.Alias.Should().Be("dbnet-v3");
    }

    [Fact]
    public void TryGetDetectionModel_WithDbNetV3Alias_ShouldReturnModel()
    {
        // Act
        var result = ModelRegistry.TryGetDetectionModel("dbnet-v3", out var modelInfo);

        // Assert
        result.Should().BeTrue();
        modelInfo.Should().NotBeNull();
        modelInfo!.RepoId.Should().Be("monkt/paddleocr-onnx");
        modelInfo.ModelFile.Should().Be("en_PP-OCRv3_det_infer.onnx");
    }

    [Fact]
    public void TryGetDetectionModel_CaseInsensitive_ShouldWork()
    {
        // Act
        var result1 = ModelRegistry.TryGetDetectionModel("DBNET-V3", out var model1);
        var result2 = ModelRegistry.TryGetDetectionModel("DbNet-V3", out var model2);
        var result3 = ModelRegistry.TryGetDetectionModel("dbnet-v3", out var model3);

        // Assert
        result1.Should().BeTrue();
        result2.Should().BeTrue();
        result3.Should().BeTrue();
        model1.Should().Be(model2);
        model2.Should().Be(model3);
    }

    [Fact]
    public void TryGetDetectionModel_WithUnknownModel_ShouldReturnFalse()
    {
        // Act
        var result = ModelRegistry.TryGetDetectionModel("unknown-model", out var modelInfo);

        // Assert
        result.Should().BeFalse();
        modelInfo.Should().BeNull();
    }

    [Fact]
    public void GetDetectionModel_WithValidAlias_ShouldReturnModel()
    {
        // Act
        var model = ModelRegistry.GetDetectionModel("dbnet-v3");

        // Assert
        model.Should().NotBeNull();
        model.Alias.Should().Be("dbnet-v3");
    }

    [Fact]
    public void GetDetectionModel_WithUnknownModel_ShouldThrow()
    {
        // Act
        var act = () => ModelRegistry.GetDetectionModel("unknown-model");

        // Assert
        act.Should().Throw<KeyNotFoundException>()
            .WithMessage("*unknown-model*not found*");
    }

    [Fact]
    public void GetAvailableDetectionModels_ShouldReturnRegisteredAliases()
    {
        // Act
        var models = ModelRegistry.GetAvailableDetectionModels().ToList();

        // Assert
        models.Should().Contain("dbnet-v3");
    }

    [Fact]
    public void DbNetV3Model_ShouldHaveCorrectConfiguration()
    {
        // Act
        var model = ModelRegistry.GetDetectionModel("dbnet-v3");

        // Assert
        model.InputWidth.Should().Be(960);
        model.InputHeight.Should().Be(960);
        model.DynamicInput.Should().BeTrue();
        model.Mean.Should().HaveCount(3);
        model.Std.Should().HaveCount(3);
    }

    #endregion

    #region Recognition Model Tests

    [Fact]
    public void TryGetRecognitionModel_WithDefaultAlias_ShouldReturnEnglishModel()
    {
        // Act
        var result = ModelRegistry.TryGetRecognitionModel("default", out var modelInfo);

        // Assert
        result.Should().BeTrue();
        modelInfo.Should().NotBeNull();
        modelInfo!.Alias.Should().Be("crnn-en-v3");
    }

    [Fact]
    public void TryGetRecognitionModel_WithEnglishAlias_ShouldReturnModel()
    {
        // Act
        var result = ModelRegistry.TryGetRecognitionModel("crnn-en-v3", out var modelInfo);

        // Assert
        result.Should().BeTrue();
        modelInfo.Should().NotBeNull();
        modelInfo!.RepoId.Should().Be("monkt/paddleocr-onnx");
        modelInfo.ModelFile.Should().Be("en_PP-OCRv3_rec_infer.onnx");
        modelInfo.DictFile.Should().Be("en_dict.txt");
    }

    [Fact]
    public void TryGetRecognitionModel_WithKoreanAlias_ShouldReturnModel()
    {
        // Act
        var result = ModelRegistry.TryGetRecognitionModel("crnn-korean-v3", out var modelInfo);

        // Assert
        result.Should().BeTrue();
        modelInfo.Should().NotBeNull();
        modelInfo!.ModelFile.Should().Be("korean_PP-OCRv3_rec_infer.onnx");
        modelInfo.LanguageCodes.Should().Contain("ko");
    }

    [Fact]
    public void TryGetRecognitionModel_CaseInsensitive_ShouldWork()
    {
        // Act
        var result1 = ModelRegistry.TryGetRecognitionModel("CRNN-EN-V3", out var model1);
        var result2 = ModelRegistry.TryGetRecognitionModel("Crnn-En-V3", out var model2);

        // Assert
        result1.Should().BeTrue();
        result2.Should().BeTrue();
        model1.Should().Be(model2);
    }

    [Fact]
    public void GetRecognitionModelForLanguage_WithEnglish_ShouldReturnEnglishModel()
    {
        // Act
        var model = ModelRegistry.GetRecognitionModelForLanguage("en");

        // Assert
        model.Alias.Should().Be("crnn-en-v3");
    }

    [Fact]
    public void GetRecognitionModelForLanguage_WithKorean_ShouldReturnKoreanModel()
    {
        // Act
        var model = ModelRegistry.GetRecognitionModelForLanguage("ko");

        // Assert
        model.Alias.Should().Be("crnn-korean-v3");
    }

    [Fact]
    public void GetRecognitionModelForLanguage_WithUnknownLanguage_ShouldFallbackToEnglish()
    {
        // Act
        var model = ModelRegistry.GetRecognitionModelForLanguage("unknown");

        // Assert
        model.Alias.Should().Be("crnn-en-v3");
    }

    [Fact]
    public void GetRecognitionModelForLanguage_WithLanguageRegion_ShouldMatch()
    {
        // Act
        var model = ModelRegistry.GetRecognitionModelForLanguage("en-US");

        // Assert
        model.Alias.Should().Be("crnn-en-v3");
    }

    [Fact]
    public void GetAvailableRecognitionModels_ShouldReturnRegisteredAliases()
    {
        // Act
        var models = ModelRegistry.GetAvailableRecognitionModels().ToList();

        // Assert
        models.Should().Contain("crnn-en-v3");
        models.Should().Contain("crnn-korean-v3");
        models.Should().Contain("crnn-chinese-v3");
        models.Should().Contain("crnn-japan-v3");
    }

    [Fact]
    public void GetSupportedLanguages_ShouldReturnLanguageCodes()
    {
        // Act
        var languages = ModelRegistry.GetSupportedLanguages().ToList();

        // Assert
        languages.Should().Contain("en");
        languages.Should().Contain("ko");
        languages.Should().Contain("zh");
        languages.Should().Contain("ja");
        languages.Should().Contain("ar");
        languages.Should().Contain("ru");
    }

    [Fact]
    public void CrnnEnV3Model_ShouldHaveCorrectConfiguration()
    {
        // Act
        var model = ModelRegistry.GetRecognitionModel("crnn-en-v3");

        // Assert
        model.InputHeight.Should().Be(48);
        model.MaxWidthRatio.Should().Be(25);
        model.UseSpace.Should().BeTrue();
        model.Mean.Should().HaveCount(3);
        model.Std.Should().HaveCount(3);
    }

    #endregion

    #region Registration Tests

    [Fact]
    public void RegisterDetectionModel_ShouldAddNewModel()
    {
        // Arrange
        var newModel = new DetectionModelInfo(
            RepoId: "test/test-detection",
            Alias: "test-detection",
            DisplayName: "Test Detection Model",
            ModelFile: "test_det.onnx");

        // Act
        ModelRegistry.RegisterDetectionModel(newModel);
        var result = ModelRegistry.TryGetDetectionModel("test-detection", out var retrieved);

        // Assert
        result.Should().BeTrue();
        retrieved.Should().Be(newModel);
    }

    [Fact]
    public void RegisterDetectionModel_WithNullModel_ShouldThrow()
    {
        // Act
        var act = () => ModelRegistry.RegisterDetectionModel(null!);

        // Assert
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void RegisterRecognitionModel_ShouldAddNewModel()
    {
        // Arrange
        var newModel = new RecognitionModelInfo(
            RepoId: "test/test-recognition",
            Alias: "test-recognition",
            DisplayName: "Test Recognition Model",
            ModelFile: "test_rec.onnx",
            DictFile: "test_dict.txt",
            LanguageCodes: ["test"]);

        // Act
        ModelRegistry.RegisterRecognitionModel(newModel);
        var result = ModelRegistry.TryGetRecognitionModel("test-recognition", out var retrieved);

        // Assert
        result.Should().BeTrue();
        retrieved.Should().Be(newModel);
    }

    [Fact]
    public void RegisterRecognitionModel_WithNullModel_ShouldThrow()
    {
        // Act
        var act = () => ModelRegistry.RegisterRecognitionModel(null!);

        // Assert
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void RegisterDetectionAlias_ShouldCreateNewMapping()
    {
        // Arrange - ensure base model exists
        ModelRegistry.TryGetDetectionModel("dbnet-v3", out _).Should().BeTrue();

        // Act
        ModelRegistry.RegisterDetectionAlias("my-det-alias", "dbnet-v3");
        var result = ModelRegistry.TryGetDetectionModel("my-det-alias", out var model);

        // Assert
        result.Should().BeTrue();
        model!.Alias.Should().Be("dbnet-v3");
    }

    [Fact]
    public void RegisterDetectionAlias_WithNonexistentModel_ShouldThrow()
    {
        // Act
        var act = () => ModelRegistry.RegisterDetectionAlias("new-alias", "nonexistent-model");

        // Assert
        act.Should().Throw<ArgumentException>()
            .WithMessage("*nonexistent-model*not found*");
    }

    [Fact]
    public void RegisterRecognitionAlias_ShouldCreateNewMapping()
    {
        // Arrange - ensure base model exists
        ModelRegistry.TryGetRecognitionModel("crnn-en-v3", out _).Should().BeTrue();

        // Act
        ModelRegistry.RegisterRecognitionAlias("my-rec-alias", "crnn-en-v3");
        var result = ModelRegistry.TryGetRecognitionModel("my-rec-alias", out var model);

        // Assert
        result.Should().BeTrue();
        model!.Alias.Should().Be("crnn-en-v3");
    }

    #endregion
}
