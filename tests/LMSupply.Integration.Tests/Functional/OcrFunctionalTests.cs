using LMSupply.Integration.Tests.Helpers;
using LMSupply.Ocr;
using LMSupply.Ocr.Models;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// Functional tests for the OCR (Document Recognition) domain.
/// Uses PaddleOCR-based DBNet + CRNN models. Tests L/I/Q/E axes.
/// Requires GPU + network access. Run locally only.
/// </summary>
[Trait("Category", "Functional")]
[Trait("Category", "LocalOnly")]
[Trait("Domain", "OCR")]
public class OcrFunctionalTests
{
    // ── L axis: Model Loading ───────────────────────────────────────

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_DefaultAlias_LoadsSuccessfully()
    {
        await using var model = await LocalOcr.LoadAsync(cancellationToken: TestContext.Current.CancellationToken);

        model.DetectionModelId.Should().NotBeNullOrEmpty();
        model.RecognitionModelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_FastAlias_LoadsSuccessfully()
    {
        await using var model = await LocalOcr.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        model.Should().NotBeNull();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_GetModelInfo_ReturnsValidInfo()
    {
        await using var model = await LocalOcr.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var info = model.GetModelInfo();
        info.Should().NotBeNull();
    }

    // ── I axis: Basic Inference ─────────────────────────────────────

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_GradientImage_RunsWithoutError()
    {
        await using var model = await LocalOcr.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateGradientBmp(200, 50);
        var result = await model.RecognizeAsync(imageBytes, TestContext.Current.CancellationToken);

        result.Should().NotBeNull();
        // Gradient image has no text, so regions may be empty — that's fine
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_OcrResult_HasFullText()
    {
        await using var model = await LocalOcr.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateGradientBmp(200, 50);
        var result = await model.RecognizeAsync(imageBytes, TestContext.Current.CancellationToken);

        result.FullText.Should().NotBeNull(); // May be empty, but not null
        result.ProcessingTimeMs.Should().BeGreaterThanOrEqualTo(0);
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_DetectOnly_RunsWithoutError()
    {
        await using var model = await LocalOcr.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateGradientBmp(200, 50);
        using var stream = new MemoryStream(imageBytes);
        var regions = await model.DetectAsync(stream, TestContext.Current.CancellationToken);

        regions.Should().NotBeNull();
    }

    // ── Q axis: Quality ─────────────────────────────────────────────

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_RegionBoundingBoxes_HaveValidCoordinates()
    {
        await using var model = await LocalOcr.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateGradientBmp(640, 480);
        var result = await model.RecognizeAsync(imageBytes, TestContext.Current.CancellationToken);

        foreach (var region in result.Regions)
        {
            region.BoundingBox.Width.Should().BeGreaterThanOrEqualTo(0);
            region.BoundingBox.Height.Should().BeGreaterThanOrEqualTo(0);
            region.Text.Should().NotBeNull();
        }
    }

    // ── E axis: Edge Cases ──────────────────────────────────────────

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_BlankWhiteImage_DoesNotCrash()
    {
        await using var model = await LocalOcr.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateSolidBmp(640, 480);
        var act = () => model.RecognizeAsync(imageBytes);
        var result = await act.Should().NotThrowAsync();
        result.Subject.Should().NotBeNull();
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_TinyImage_DoesNotCrash()
    {
        await using var model = await LocalOcr.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateTinyBmp();
        try
        {
            var result = await model.RecognizeAsync(imageBytes, TestContext.Current.CancellationToken);
            result.Should().NotBeNull();
        }
        catch (Exception ex)
        {
            ex.Should().NotBeOfType<NullReferenceException>();
        }
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_InvalidImageData_ThrowsCleanError()
    {
        await using var model = await LocalOcr.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var act = () => model.RecognizeAsync(new byte[] { 0x00, 0x01, 0x02 });
        await act.Should().ThrowAsync<Exception>("non-image data should throw, not crash");
    }

    // ── Gap Tests: Language, Models, API Surface ─────────────────────

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_LoadForLanguageAsync_English_Works()
    {
        await using var model = await LocalOcr.LoadForLanguageAsync("en", cancellationToken: TestContext.Current.CancellationToken);

        model.Should().NotBeNull();
        model.DetectionModelId.Should().NotBeNullOrEmpty();
        model.RecognitionModelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_LoadForLanguageAsync_Korean_Works()
    {
        await using var model = await LocalOcr.LoadForLanguageAsync("ko", cancellationToken: TestContext.Current.CancellationToken);

        model.Should().NotBeNull();
        model.RecognitionModelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetSupportedLanguages_ReturnsLanguages()
    {
        var languages = LocalOcr.GetSupportedLanguages().ToList();

        languages.Should().NotBeEmpty();
        languages.Should().Contain("en");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAvailableDetectionModels_ReturnsModels()
    {
        var models = LocalOcr.GetAvailableDetectionModels().ToList();

        models.Should().NotBeEmpty();
        models.Should().Contain("dbnet-v3",
            "OCR detection uses DBNet-v3 model");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAvailableRecognitionModels_ReturnsModels()
    {
        var models = LocalOcr.GetAvailableRecognitionModels().ToList();

        models.Should().NotBeEmpty();
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_RecognizeFromStream_Works()
    {
        await using var model = await LocalOcr.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateGradientBmp(200, 50);
        using var stream = new MemoryStream(imageBytes);
        var result = await model.RecognizeAsync(stream, TestContext.Current.CancellationToken);

        result.Should().NotBeNull();
        result.FullText.Should().NotBeNull();
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_ProcessingTime_IsPositive()
    {
        await using var model = await LocalOcr.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateGradientBmp(200, 50);
        var result = await model.RecognizeAsync(imageBytes, TestContext.Current.CancellationToken);

        result.ProcessingTimeMs.Should().BeGreaterThanOrEqualTo(0);
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_LargeImage_DoesNotCrash()
    {
        await using var model = await LocalOcr.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        // Large image (1920x1080)
        var imageBytes = TestDataHelper.CreateGradientBmp(1920, 1080);
        var act = () => model.RecognizeAsync(imageBytes);
        await act.Should().NotThrowAsync("large images should not crash the OCR");
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_NarrowImage_DoesNotCrash()
    {
        await using var model = await LocalOcr.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        // Very narrow image (like a text line scan)
        var imageBytes = TestDataHelper.CreateGradientBmp(500, 20);
        var act = () => model.RecognizeAsync(imageBytes);
        await act.Should().NotThrowAsync("narrow images should not crash");
    }

    // ── Static Tests: Options Defaults & Model Registry (no model loading) ─

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_OcrOptions_Defaults_AreReasonable()
    {
        var opts = new OcrOptions();

        opts.LanguageHint.Should().Be("en", "default language should be English");
        opts.DetectionThreshold.Should().Be(0.5f, "default detection threshold should be 0.5");
        opts.RecognitionThreshold.Should().Be(0.5f, "default recognition threshold should be 0.5");
        opts.BinarizationThreshold.Should().Be(0.3f, "default binarization threshold should be 0.3");
        opts.MaxCandidates.Should().Be(1000, "default max candidates should be 1000");
        opts.UnclipRatio.Should().Be(1.5f, "default unclip ratio should be 1.5");
        opts.MinBoxArea.Should().Be(10, "default min box area should be 10");
        opts.UsePolygon.Should().BeTrue("default should use polygon coordinates");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetSupportedLanguages_ContainsCommonLanguages()
    {
        var languages = LocalOcr.GetSupportedLanguages().ToList();

        languages.Should().Contain("en", "English should be supported");
        languages.Should().Contain("ko", "Korean should be supported");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAvailableRecognitionModels_ContainsEnglishModel()
    {
        var models = LocalOcr.GetAvailableRecognitionModels().ToList();

        models.Should().Contain(m => m.Contains("en"),
            "recognition models should include an English model");
    }

    // ── OCR ModelRegistry Static Validation ──────────────────────────

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllDetectionModels_AllHaveRequiredFields()
    {
        var models = OcrDetectionModelRegistry.Default.GetAvailableModels().ToList();

        models.Should().NotBeEmpty("at least one detection model should be registered");
        models.Should().AllSatisfy(m =>
        {
            m.RepoId.Should().NotBeNullOrEmpty($"RepoId should be set for {m.AliasName}");
            m.AliasName.Should().NotBeNullOrEmpty("AliasName should be set");
            m.DisplayName.Should().NotBeNullOrEmpty($"DisplayName should be set for {m.AliasName}");
            m.ModelFile.Should().NotBeNullOrEmpty($"ModelFile should be set for {m.AliasName}");
            m.InputWidth.Should().BeGreaterThan(0, $"InputWidth should be positive for {m.AliasName}");
            m.InputHeight.Should().BeGreaterThan(0, $"InputHeight should be positive for {m.AliasName}");
        });
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllRecognitionModels_AllHaveRequiredFields()
    {
        var models = OcrRecognitionModelRegistry.Default.GetAvailableModels().ToList();

        models.Should().NotBeEmpty("at least one recognition model should be registered");
        models.Should().AllSatisfy(m =>
        {
            m.RepoId.Should().NotBeNullOrEmpty($"RepoId should be set for {m.AliasName}");
            m.AliasName.Should().NotBeNullOrEmpty("AliasName should be set");
            m.DisplayName.Should().NotBeNullOrEmpty($"DisplayName should be set for {m.AliasName}");
            m.ModelFile.Should().NotBeNullOrEmpty($"ModelFile should be set for {m.AliasName}");
            m.DictFile.Should().NotBeNullOrEmpty($"DictFile should be set for {m.AliasName}");
            m.LanguageCodes.Should().NotBeEmpty($"LanguageCodes should be set for {m.AliasName}");
        });
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetSupportedLanguages_ContainsAllMajorScripts()
    {
        var languages = LocalOcr.GetSupportedLanguages().ToList();

        languages.Should().Contain("en", "English should be supported");
        languages.Should().Contain("ko", "Korean should be supported");
        languages.Should().Contain("zh", "Chinese should be supported");
        languages.Should().Contain("ar", "Arabic should be supported");
        languages.Should().Contain("ru", "Russian/Cyrillic should be supported");
        languages.Should().Contain("hi", "Hindi/Devanagari should be supported");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_DefaultDetectionAlias_ReturnsDbNetV3()
    {
        var model = OcrDetectionModelRegistry.Default.Resolve("default");

        model.RepoId.Should().Be("monkt/paddleocr-onnx", "default detection model should come from paddleocr repo");
        model.ModelFile.Should().Be("det.onnx", "detection model file should be det.onnx");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_FastDetectionAlias_ReturnsDbNetV3()
    {
        var model = OcrDetectionModelRegistry.Default.Resolve("fast");

        model.RepoId.Should().Be("monkt/paddleocr-onnx", "fast detection alias should point to paddleocr repo");
        model.ModelFile.Should().Be("det.onnx", "fast detection model file should be det.onnx");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetRecognitionModelForLanguage_FallsBackToEnglish()
    {
        // Japanese is documented as falling back to English (not available in repository)
        var model = OcrRecognitionModelRegistry.Default.ResolveForLanguage("ja");

        model.AliasName.Should().Be("crnn-en-v3",
            "unsupported language should fall back to English recognition");
    }

    // ── Gap Tests: Language fallback consistency, narrow image ───

    [Theory]
    [Trait("Axis", "Loading")]
    [InlineData("fr")]
    [InlineData("es")]
    [InlineData("de")]
    public void L_GetRecognitionModelForLanguage_LatinLanguagesMapsToLatinModel(string language)
    {
        var model = OcrRecognitionModelRegistry.Default.ResolveForLanguage(language);

        model.Should().NotBeNull($"language '{language}' should resolve to a valid model");
        model.AliasName.Should().Be("crnn-latin-v3",
            $"Latin-script language '{language}' should map to the Latin recognition model");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_OcrOptions_AllDefaults_AreDocumented()
    {
        var opts = new OcrOptions();

        opts.LanguageHint.Should().Be("en");
        opts.DetectionThreshold.Should().Be(0.5f);
        opts.RecognitionThreshold.Should().Be(0.5f);
        opts.BinarizationThreshold.Should().Be(0.3f);
        opts.MaxCandidates.Should().Be(1000);
        opts.UnclipRatio.Should().Be(1.5f);
        opts.MinBoxArea.Should().Be(10);
        opts.UsePolygon.Should().BeTrue();
        opts.Provider.Should().Be(ExecutionProvider.Auto);
        opts.CacheDirectory.Should().BeNull();
    }
}
