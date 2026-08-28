using AwesomeAssertions;
using LMSupply.Ocr.Models;

namespace LMSupply.Ocr.Tests;

public class ModelRegistryTests
{
    #region Detection Registry Tests

    [Fact]
    public void DetectionRegistry_Resolve_Default_ShouldReturnDbNetV3()
    {
        // Act
        var model = OcrDetectionModelRegistry.Default.Resolve("default");

        // Assert
        model.Should().NotBeNull();
        model.RepoId.Should().Be("monkt/paddleocr-onnx");
        model.ModelFile.Should().Be("det.onnx");
    }

    [Fact]
    public void DetectionRegistry_Resolve_DbNetV3_ShouldReturnModel()
    {
        // Act
        var model = OcrDetectionModelRegistry.Default.Resolve("dbnet-v3");

        // Assert
        model.Should().NotBeNull();
        model.RepoId.Should().Be("monkt/paddleocr-onnx");
        model.ModelFile.Should().Be("det.onnx");
        model.Subfolder.Should().Be("detection/v3");
    }

    [Fact]
    public void DetectionRegistry_Resolve_FastAlias_ShouldReturnDbNetV3()
    {
        // Act — "fast" alias added in cycle 135 (single-model domain)
        var model = OcrDetectionModelRegistry.Default.Resolve("fast");

        // Assert
        model.Should().NotBeNull();
        model.RepoId.Should().Be("monkt/paddleocr-onnx");
        model.ModelFile.Should().Be("det.onnx");
    }

    [Fact]
    public void DetectionRegistry_TryResolve_CaseInsensitive_ShouldWork()
    {
        // Act
        var result1 = OcrDetectionModelRegistry.Default.TryResolve("DBNET-V3", out var model1);
        var result2 = OcrDetectionModelRegistry.Default.TryResolve("DbNet-V3", out var model2);
        var result3 = OcrDetectionModelRegistry.Default.TryResolve("dbnet-v3", out var model3);

        // Assert
        result1.Should().BeTrue();
        result2.Should().BeTrue();
        result3.Should().BeTrue();
        model1!.RepoId.Should().Be(model2!.RepoId);
        model2!.RepoId.Should().Be(model3!.RepoId);
    }

    [Fact]
    public void DetectionRegistry_TryResolve_UnknownModel_ShouldReturnFalse()
    {
        // Act
        var result = OcrDetectionModelRegistry.Default.TryResolve("unknown-model", out var modelInfo);

        // Assert
        result.Should().BeFalse();
        modelInfo.Should().BeNull();
    }

    [Fact]
    public void DetectionRegistry_Resolve_UnknownModel_ShouldThrow()
    {
        // Act
        var act = () => OcrDetectionModelRegistry.Default.Resolve("unknown-model");

        // Assert
        act.Should().Throw<Exception>()
            .WithMessage("*unknown-model*not found*");
    }

    [Fact]
    public void DetectionRegistry_GetAvailableModels_ShouldContainDbNetV3()
    {
        // Act
        var models = OcrDetectionModelRegistry.Default.GetAvailableModels();

        // Assert
        models.Should().Contain(m => m.AliasName == "default" || m.AliasName == "dbnet-v3");
    }

    [Fact]
    public void DetectionRegistry_Auto_ShouldResolve()
    {
        // Act
        var model = OcrDetectionModelRegistry.Default.Resolve("auto");

        // Assert
        model.Should().NotBeNull();
        model.AliasName.Should().Be("auto");
        model.RepoId.Should().Be("monkt/paddleocr-onnx");
    }

    [Fact]
    public void DbNetV3Model_ShouldHaveCorrectConfiguration()
    {
        // Act
        var model = OcrDetectionModelRegistry.Default.Resolve("dbnet-v3");

        // Assert
        model.InputWidth.Should().Be(960);
        model.InputHeight.Should().Be(960);
        model.DynamicInput.Should().BeTrue();
        model.Mean.Should().HaveCount(3);
        model.Std.Should().HaveCount(3);
    }

    #endregion

    #region Recognition Registry Tests

    [Fact]
    public void RecognitionRegistry_Resolve_Default_ShouldReturnEnglishModel()
    {
        // Act
        var model = OcrRecognitionModelRegistry.Default.Resolve("default");

        // Assert
        model.Should().NotBeNull();
        model.LanguageCodes.Should().Contain("en");
        model.RepoId.Should().Be("monkt/paddleocr-onnx");
    }

    [Fact]
    public void RecognitionRegistry_Resolve_CrnnEnV3_ShouldReturnModel()
    {
        // Act
        var model = OcrRecognitionModelRegistry.Default.Resolve("crnn-en-v3");

        // Assert
        model.Should().NotBeNull();
        model.RepoId.Should().Be("monkt/paddleocr-onnx");
        model.ModelFile.Should().Be("rec.onnx");
        model.DictFile.Should().Be("dict.txt");
        model.Subfolder.Should().Be("languages/english");
    }

    [Fact]
    public void RecognitionRegistry_Resolve_KoreanAlias_ShouldReturnModel()
    {
        // Act
        var model = OcrRecognitionModelRegistry.Default.Resolve("crnn-korean-v3");

        // Assert
        model.Should().NotBeNull();
        model.ModelFile.Should().Be("rec.onnx");
        model.Subfolder.Should().Be("languages/korean");
        model.LanguageCodes.Should().Contain("ko");
    }

    [Fact]
    public void RecognitionRegistry_TryResolve_CaseInsensitive_ShouldWork()
    {
        // Act
        var result1 = OcrRecognitionModelRegistry.Default.TryResolve("CRNN-EN-V3", out var model1);
        var result2 = OcrRecognitionModelRegistry.Default.TryResolve("Crnn-En-V3", out var model2);

        // Assert
        result1.Should().BeTrue();
        result2.Should().BeTrue();
        model1!.RepoId.Should().Be(model2!.RepoId);
    }

    [Fact]
    public void RecognitionRegistry_ResolveForLanguage_English_ShouldReturnEnglishModel()
    {
        // Act
        var model = OcrRecognitionModelRegistry.Default.ResolveForLanguage("en");

        // Assert
        model.LanguageCodes.Should().Contain("en");
    }

    [Fact]
    public void RecognitionRegistry_ResolveForLanguage_Korean_ShouldReturnKoreanModel()
    {
        // Act
        var model = OcrRecognitionModelRegistry.Default.ResolveForLanguage("ko");

        // Assert
        model.LanguageCodes.Should().Contain("ko");
    }

    [Fact]
    public void RecognitionRegistry_ResolveForLanguage_UnknownLanguage_ShouldFallbackToEnglish()
    {
        // Act
        var model = OcrRecognitionModelRegistry.Default.ResolveForLanguage("unknown");

        // Assert
        model.LanguageCodes.Should().Contain("en");
    }

    [Fact]
    public void RecognitionRegistry_ResolveForLanguage_WithRegion_ShouldMatch()
    {
        // Act
        var model = OcrRecognitionModelRegistry.Default.ResolveForLanguage("en-US");

        // Assert
        model.LanguageCodes.Should().Contain("en");
    }

    [Fact]
    public void RecognitionRegistry_LanguageCode_ShouldResolveDirectly()
    {
        // Language codes are registered as system aliases
        // Act
        var enModel = OcrRecognitionModelRegistry.Default.Resolve("en");
        var koModel = OcrRecognitionModelRegistry.Default.Resolve("ko");
        var zhModel = OcrRecognitionModelRegistry.Default.Resolve("zh");
        var arModel = OcrRecognitionModelRegistry.Default.Resolve("ar");
        var ruModel = OcrRecognitionModelRegistry.Default.Resolve("ru");

        // Assert
        enModel.LanguageCodes.Should().Contain("en");
        koModel.LanguageCodes.Should().Contain("ko");
        zhModel.LanguageCodes.Should().Contain("zh");
        arModel.LanguageCodes.Should().Contain("ar");
        ruModel.LanguageCodes.Should().Contain("ru");
    }

    [Fact]
    public void RecognitionRegistry_GetAvailableModels_ShouldReturnRegisteredModels()
    {
        // Act
        var models = OcrRecognitionModelRegistry.Default.GetAvailableModels();

        // Assert (distinct by RepoId — deduplicated by base class)
        models.Should().NotBeEmpty();
    }

    [Fact]
    public void RecognitionRegistry_GetSupportedLanguages_ShouldReturnLanguageCodes()
    {
        // Act
        var languages = OcrRecognitionModelRegistry.Default.GetSupportedLanguages().ToList();

        // Assert
        languages.Should().Contain("en");
        languages.Should().Contain("ko");
        languages.Should().Contain("zh");
        languages.Should().Contain("ar");
        languages.Should().Contain("ru");
    }

    [Fact]
    public void RecognitionRegistry_Auto_ShouldResolve()
    {
        // Act
        var model = OcrRecognitionModelRegistry.Default.Resolve("auto");

        // Assert
        model.Should().NotBeNull();
        model.AliasName.Should().Be("auto");
        model.LanguageCodes.Should().Contain("en");
    }

    [Fact]
    public void CrnnEnV3Model_ShouldHaveCorrectConfiguration()
    {
        // Act
        var model = OcrRecognitionModelRegistry.Default.Resolve("crnn-en-v3");

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
    public void DetectionRegistry_RegisterAlias_ShouldCreateNewMapping()
    {
        // Arrange
        var registry = new OcrDetectionModelRegistry(DefaultDetectionModels.All);

        // Act
        registry.RegisterAlias("my-det-alias", "dbnet-v3");
        var result = registry.TryResolve("my-det-alias", out var model);

        // Assert
        result.Should().BeTrue();
        model!.RepoId.Should().Be("monkt/paddleocr-onnx");
    }

    [Fact]
    public void RecognitionRegistry_RegisterAlias_ShouldCreateNewMapping()
    {
        // Arrange
        var registry = new OcrRecognitionModelRegistry(DefaultRecognitionModels.All);

        // Act
        registry.RegisterAlias("my-rec-alias", "crnn-en-v3");
        var result = registry.TryResolve("my-rec-alias", out var model);

        // Assert
        result.Should().BeTrue();
        model!.LanguageCodes.Should().Contain("en");
    }

    [Fact]
    public void DetectionRegistry_GetAliases_ShouldIncludeSystemAliases()
    {
        // Act
        var aliases = OcrDetectionModelRegistry.Default.GetAliases();

        // Assert
        aliases.Should().Contain(a => a.Name == "auto");
        aliases.Should().Contain(a => a.Name == "default");
        aliases.Should().Contain(a => a.Name == "fast");
        aliases.Should().Contain(a => a.Name == "dbnet-v3");
    }

    [Fact]
    public void DefaultDetectionModels_All_ShouldContainAllModels()
    {
        // Assert
        DefaultDetectionModels.All.Should().HaveCount(3);
        DefaultDetectionModels.All.Should().Contain(DefaultDetectionModels.DbNetV3);
        DefaultDetectionModels.All.Should().Contain(DefaultDetectionModels.DbNetV3Default);
        DefaultDetectionModels.All.Should().Contain(DefaultDetectionModels.DbNetV3Fast);
    }

    [Fact]
    public void RecognitionRegistry_GetAliases_ShouldIncludeLanguageCodeAliases()
    {
        // Act
        var aliases = OcrRecognitionModelRegistry.Default.GetAliases();

        // Assert
        aliases.Should().Contain(a => a.Name == "auto");
        aliases.Should().Contain(a => a.Name == "default");
        aliases.Should().Contain(a => a.Name == "crnn-en-v3");
        aliases.Should().Contain(a => a.Name == "en");
        aliases.Should().Contain(a => a.Name == "ko");
        aliases.Should().Contain(a => a.Name == "zh");
    }

    #endregion

    #region LocalOcr Registry Exposure Tests

    [Fact]
    public void LocalOcr_DetectionRegistry_ShouldExposeDefault()
    {
        // Act
        var registry = LocalOcr.DetectionRegistry;

        // Assert
        registry.Should().NotBeNull();
        registry.Should().BeSameAs(OcrDetectionModelRegistry.Default);
    }

    [Fact]
    public void LocalOcr_RecognitionRegistry_ShouldExposeDefault()
    {
        // Act
        var registry = LocalOcr.RecognitionRegistry;

        // Assert
        registry.Should().NotBeNull();
        registry.Should().BeSameAs(OcrRecognitionModelRegistry.Default);
    }

    #endregion

}
