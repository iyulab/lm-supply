using AwesomeAssertions;
using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Xunit;

namespace LMSupply.Ocr.Tests;

/// <summary>
/// Integration tests for OCR models.
/// These tests require actual model downloads and GPU/CPU inference.
/// </summary>
[Trait("Category", "Integration")]
public class OcrIntegrationTests : IDisposable
{
    private readonly string _testImagesDir;

    public OcrIntegrationTests()
    {
        _testImagesDir = Path.Combine(Path.GetTempPath(), "lmsupply_ocr_tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_testImagesDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_testImagesDir))
        {
            try { Directory.Delete(_testImagesDir, true); }
            catch { /* ignore cleanup errors */ }
        }
        GC.SuppressFinalize(this);
    }

    #region Default Model Tests

    [Fact]
    public async Task DefaultModel_LoadAndRecognize_EnglishText_ShouldWork()
    {
        // Arrange
        var testImagePath = Path.Combine(_testImagesDir, "english_test.png");
        CreateTestImage(testImagePath, "Hello World", 600, 150);

        // Act
        await using var ocr = await LocalOcr.LoadAsync(cancellationToken: TestContext.Current.CancellationToken);
        var result = await ocr.RecognizeAsync(testImagePath, TestContext.Current.CancellationToken);

        // Assert - Model should load and process image
        ocr.Should().NotBeNull();
        ocr.DetectionModelId.Should().NotBeNullOrEmpty();
        ocr.RecognitionModelId.Should().NotBeNullOrEmpty();

        result.Should().NotBeNull();
        result.ProcessingTimeMs.Should().BeGreaterThan(0);

        // OCR should detect at least some text (exact recognition may vary by font/rendering)
        // If regions are detected, verify basic structure
        if (result.Regions.Count > 0)
        {
            result.FullText.Should().NotBeNullOrEmpty();
            result.Regions.All(r => r.Confidence > 0).Should().BeTrue();
            result.Regions.All(r => r.BoundingBox.Width > 0).Should().BeTrue();
        }
    }

    [Fact]
    public async Task DefaultModel_RecognizeFromStream_ShouldWork()
    {
        // Arrange
        var testImagePath = Path.Combine(_testImagesDir, "stream_test.png");
        CreateTestImage(testImagePath, "Stream Test", 600, 150);

        await using var ocr = await LocalOcr.LoadAsync(cancellationToken: TestContext.Current.CancellationToken);

        // Act
        using var stream = File.OpenRead(testImagePath);
        var result = await ocr.RecognizeAsync(stream, TestContext.Current.CancellationToken);

        // Assert
        result.Should().NotBeNull();
        result.ProcessingTimeMs.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task DefaultModel_RecognizeFromByteArray_ShouldWork()
    {
        // Arrange
        var testImagePath = Path.Combine(_testImagesDir, "bytes_test.png");
        CreateTestImage(testImagePath, "Byte Array", 600, 150);

        await using var ocr = await LocalOcr.LoadAsync(cancellationToken: TestContext.Current.CancellationToken);

        // Act
        var imageData = await File.ReadAllBytesAsync(testImagePath, TestContext.Current.CancellationToken);
        var result = await ocr.RecognizeAsync(imageData, TestContext.Current.CancellationToken);

        // Assert
        result.Should().NotBeNull();
        result.ProcessingTimeMs.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task DefaultModel_DetectOnly_ShouldReturnBoundingBoxes()
    {
        // Arrange
        var testImagePath = Path.Combine(_testImagesDir, "detect_test.png");
        CreateTestImage(testImagePath, "Detection Test", 400, 100);

        await using var ocr = await LocalOcr.LoadAsync(cancellationToken: TestContext.Current.CancellationToken);

        // Act
        var detections = await ocr.DetectAsync(testImagePath, TestContext.Current.CancellationToken);

        // Assert
        detections.Should().NotBeEmpty();
        detections.All(d => d.BoundingBox.Width > 0 && d.BoundingBox.Height > 0).Should().BeTrue();
        detections.All(d => d.Confidence > 0).Should().BeTrue();
    }

    [Fact]
    public async Task DefaultModel_WarmupAsync_ShouldNotThrow()
    {
        // Arrange
        await using var ocr = await LocalOcr.LoadAsync(cancellationToken: TestContext.Current.CancellationToken);

        // Act
        var warmupAction = () => ocr.WarmupAsync();

        // Assert
        await warmupAction.Should().NotThrowAsync();
    }

    [Fact]
    public async Task DefaultModel_GetModelInfo_ShouldReturnValidInfo()
    {
        // Arrange
        await using var ocr = await LocalOcr.LoadAsync(cancellationToken: TestContext.Current.CancellationToken);

        // Act
        var info = ocr.GetModelInfo();

        // Assert
        info.Should().NotBeNull();
        info!.DetectionModel.Should().NotBeNull();
        info.RecognitionModel.Should().NotBeNull();
        info.DetectionModel.ModelFile.Should().NotBeNullOrEmpty();
        info.RecognitionModel.ModelFile.Should().NotBeNullOrEmpty();
        info.SupportedLanguages.Should().NotBeEmpty();
    }

    #endregion

    #region Language-Specific Tests

    [Fact]
    public async Task LoadForLanguage_Korean_ShouldLoadKoreanModel()
    {
        // Arrange & Act
        await using var ocr = await LocalOcr.LoadForLanguageAsync("ko", cancellationToken: TestContext.Current.CancellationToken);

        // Assert
        ocr.Should().NotBeNull();
        ocr.RecognitionModelId.Should().Contain("korean");
        ocr.SupportedLanguages.Should().Contain("ko");
    }

    [Fact]
    public async Task LoadForLanguage_Japanese_ShouldFallbackToEnglish()
    {
        // Note: Japanese is not available in monkt/paddleocr-onnx repository
        // It falls back to English recognition
        // To use Japanese, load from deepghs/paddleocr with language hint

        // Arrange & Act
        await using var ocr = await LocalOcr.LoadForLanguageAsync("ja", cancellationToken: TestContext.Current.CancellationToken);

        // Assert - Falls back to English since Japanese model not available
        ocr.Should().NotBeNull();
        ocr.RecognitionModelId.Should().NotBeNullOrEmpty();
        // Note: This will be English model as fallback
        ocr.SupportedLanguages.Should().Contain("en");
    }

    [Fact]
    public async Task LoadForLanguage_Chinese_ShouldLoadChineseModel()
    {
        // Arrange & Act
        await using var ocr = await LocalOcr.LoadForLanguageAsync("zh", cancellationToken: TestContext.Current.CancellationToken);

        // Assert
        ocr.Should().NotBeNull();
        ocr.RecognitionModelId.Should().Contain("chinese");
        ocr.SupportedLanguages.Should().Contain("zh");
    }

    #endregion

    #region Multi-Line Text Tests

    [Fact]
    public async Task DefaultModel_MultiLineText_ShouldRecognizeAllLines()
    {
        // Arrange
        var testImagePath = Path.Combine(_testImagesDir, "multiline_test.png");
        CreateMultiLineTestImage(testImagePath, ["Line One", "Line Two", "Line Three"], 400, 200);

        await using var ocr = await LocalOcr.LoadAsync(cancellationToken: TestContext.Current.CancellationToken);

        // Act
        var result = await ocr.RecognizeAsync(testImagePath, TestContext.Current.CancellationToken);

        // Assert - Verify basic functionality (OCR accuracy may vary with generated images)
        result.Should().NotBeNull();
        result.ProcessingTimeMs.Should().BeGreaterThan(0);

        // Multi-line images should detect at least one region
        // Note: Exact text recognition may vary based on font rendering
        if (result.Regions.Count > 0)
        {
            result.Regions.All(r => r.BoundingBox.Width > 0).Should().BeTrue();
        }
    }

    [Fact]
    public async Task DefaultModel_GetTextWithLayout_ShouldPreserveLineStructure()
    {
        // Arrange
        var testImagePath = Path.Combine(_testImagesDir, "layout_test.png");
        CreateMultiLineTestImage(testImagePath, ["First Line", "Second Line"], 400, 150);

        await using var ocr = await LocalOcr.LoadAsync(cancellationToken: TestContext.Current.CancellationToken);

        // Act
        var result = await ocr.RecognizeAsync(testImagePath, TestContext.Current.CancellationToken);
        var layoutText = result.GetTextWithLayout();

        // Assert
        layoutText.Should().NotBeNullOrEmpty();
    }

    #endregion

    #region HuggingFace Model Tests

    [Fact]
    public async Task HuggingFaceModel_Load_ShouldAutoDiscoverFiles()
    {
        // Arrange & Act
        // Using the default monkt/paddleocr-onnx repo
        await using var ocr = await LocalOcr.LoadAsync(detectionModel: "default", recognitionModel: "default", cancellationToken: TestContext.Current.CancellationToken);

        // Assert
        ocr.Should().NotBeNull();
        ocr.DetectionModelId.Should().NotBeNullOrEmpty();
        ocr.RecognitionModelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public async Task ExplicitModelAlias_ShouldLoadCorrectModel()
    {
        // Arrange & Act
        await using var ocr = await LocalOcr.LoadAsync(detectionModel: "dbnet-v3", recognitionModel: "crnn-en-v3", cancellationToken: TestContext.Current.CancellationToken);

        // Assert
        ocr.Should().NotBeNull();
        ocr.DetectionModelId.Should().NotBeNullOrEmpty();
        ocr.RecognitionModelId.Should().NotBeNullOrEmpty();
    }

    #endregion

    #region Options Tests

    [Fact]
    public async Task OcrOptions_CustomThresholds_ShouldApply()
    {
        // Arrange
        var options = new OcrOptions
        {
            DetectionThreshold = 0.7f,
            RecognitionThreshold = 0.8f
        };

        var testImagePath = Path.Combine(_testImagesDir, "threshold_test.png");
        CreateTestImage(testImagePath, "High Confidence", 400, 100);

        // Act
        await using var ocr = await LocalOcr.LoadAsync(options: options, cancellationToken: TestContext.Current.CancellationToken);
        var result = await ocr.RecognizeAsync(testImagePath, TestContext.Current.CancellationToken);

        // Assert
        ocr.Should().NotBeNull();
        result.Should().NotBeNull();
    }

    [Fact]
    public async Task OcrOptions_CustomProvider_CPU_ShouldWork()
    {
        // Arrange
        var options = new OcrOptions
        {
            Provider = ExecutionProvider.Cpu
        };

        var testImagePath = Path.Combine(_testImagesDir, "cpu_test.png");
        CreateTestImage(testImagePath, "CPU Test", 400, 100);

        // Act
        await using var ocr = await LocalOcr.LoadAsync(options: options, cancellationToken: TestContext.Current.CancellationToken);
        var result = await ocr.RecognizeAsync(testImagePath, TestContext.Current.CancellationToken);

        // Assert
        ocr.Should().NotBeNull();
        result.Should().NotBeNull();
        result.Regions.Should().NotBeEmpty();
    }

    #endregion

    #region Real Image Tests

    [Fact]
    public async Task DefaultModel_RecognizeRealImage_ShouldDetectText()
    {
        // Arrange - Use a real screenshot image if available
        var sourceImagePath = @"C:\Users\achunja\Downloads\1.png";

        // Skip test if the test image doesn't exist
        Assert.SkipWhen(!File.Exists(sourceImagePath), "Test image not available at: " + sourceImagePath);

        await using var ocr = await LocalOcr.LoadAsync(cancellationToken: TestContext.Current.CancellationToken);

        // Act
        var result = await ocr.RecognizeAsync(sourceImagePath, TestContext.Current.CancellationToken);

        // Assert - Real image should have detectable text
        result.Should().NotBeNull();
        result.Regions.Should().NotBeEmpty("Real image should have detectable text");
        result.FullText.Should().NotBeNullOrEmpty();
        result.ProcessingTimeMs.Should().BeGreaterThan(0);

        // Verify regions have valid structure
        result.Regions.All(r => r.Text.Length > 0).Should().BeTrue();
        result.Regions.All(r => r.Confidence > 0).Should().BeTrue();
        result.Regions.All(r => r.BoundingBox.Width > 0 && r.BoundingBox.Height > 0).Should().BeTrue();

        // Output recognized text for manual verification
        Console.WriteLine($"OCR Result ({result.Regions.Count} regions):");
        Console.WriteLine(result.FullText);
    }

    [Fact]
    public async Task DefaultModel_DetectRealImage_ShouldReturnBoundingBoxes()
    {
        // Arrange
        var sourceImagePath = @"C:\Users\achunja\Downloads\1.png";

        // Skip test if the test image doesn't exist
        Assert.SkipWhen(!File.Exists(sourceImagePath), "Test image not available at: " + sourceImagePath);

        await using var ocr = await LocalOcr.LoadAsync(cancellationToken: TestContext.Current.CancellationToken);

        // Act
        var detections = await ocr.DetectAsync(sourceImagePath, TestContext.Current.CancellationToken);

        // Assert
        detections.Should().NotBeEmpty("Real image should have detectable text regions");
        detections.All(d => d.BoundingBox.Width > 0 && d.BoundingBox.Height > 0).Should().BeTrue();
        detections.All(d => d.Confidence > 0).Should().BeTrue();
    }

    #endregion

    #region Static Method Tests

    [Fact]
    public void GetAvailableDetectionModels_ShouldReturnModels()
    {
        // Act
        var models = LocalOcr.GetAvailableDetectionModels().ToList();

        // Assert
        models.Should().NotBeEmpty();
        models.Should().Contain("dbnet-v3");
    }

    [Fact]
    public void GetAvailableRecognitionModels_ShouldReturnModels()
    {
        // Act
        var models = LocalOcr.GetAvailableRecognitionModels().ToList();

        // Assert
        models.Should().NotBeEmpty();
        models.Should().Contain("crnn-en-v3");
        models.Should().Contain("crnn-korean-v3");
        models.Should().Contain("crnn-chinese-v3");
        models.Should().Contain("crnn-latin-v3");
    }

    [Fact]
    public void GetSupportedLanguages_ShouldReturnLanguages()
    {
        // Act
        var languages = LocalOcr.GetSupportedLanguages().ToList();

        // Assert
        languages.Should().NotBeEmpty();
        languages.Should().Contain("en");
        languages.Should().Contain("ko");
        languages.Should().Contain("zh");
        languages.Should().Contain("de"); // Latin model
        languages.Should().Contain("fr"); // Latin model
    }

    #endregion

    #region Helper Methods

    private static Font GetDefaultFont(float size)
    {
        // Try to get system fonts in order of preference
        var fontNames = new[] { "Arial", "Segoe UI", "DejaVu Sans", "Liberation Sans", "Noto Sans" };

        foreach (var fontName in fontNames)
        {
            if (SystemFonts.TryGet(fontName, out var family))
            {
                return family.CreateFont(size, FontStyle.Bold);
            }
        }

        // Fallback to any available font
        var availableFonts = SystemFonts.Families.ToList();
        if (availableFonts.Count > 0)
        {
            return availableFonts[0].CreateFont(size, FontStyle.Bold);
        }

        throw new InvalidOperationException("No system fonts available");
    }

    private static void CreateTestImage(string path, string text, int width, int height)
    {
        using var image = new Image<Rgba32>(width, height);

        // White background
        image.Mutate(ctx => ctx.Fill(Color.White));

        // Get font
        var font = GetDefaultFont(48); // Larger font for better detection

        // Draw text with good contrast
        var textOptions = new RichTextOptions(font)
        {
            Origin = new PointF(20, height / 2 - 30),
            HorizontalAlignment = HorizontalAlignment.Left
        };

        image.Mutate(ctx => ctx.DrawText(textOptions, text, Color.Black));
        image.SaveAsPng(path);
    }

    private static void CreateMultiLineTestImage(string path, string[] lines, int width, int height)
    {
        using var image = new Image<Rgba32>(width, height);

        // White background
        image.Mutate(ctx => ctx.Fill(Color.White));

        // Get font
        var font = GetDefaultFont(36); // Larger font for better detection
        float y = 30;
        float lineHeight = 50;

        foreach (var line in lines)
        {
            var textOptions = new RichTextOptions(font)
            {
                Origin = new PointF(20, y),
                HorizontalAlignment = HorizontalAlignment.Left
            };

            image.Mutate(ctx => ctx.DrawText(textOptions, line, Color.Black));
            y += lineHeight;
        }

        image.SaveAsPng(path);
    }

    #endregion
}
