using LMSupply.Detector;
using LMSupply.Integration.Tests.Helpers;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// Functional tests for the Detector (Object Detection) domain.
/// Uses RT-DETR v2 models. Tests L/I/Q/E axes.
/// Requires GPU + network access. Run locally only.
/// </summary>
[Trait("Category", "Functional")]
[Trait("Category", "LocalOnly")]
[Trait("Domain", "Detector")]
public class DetectorFunctionalTests
{
    // ── L axis: Model Loading ───────────────────────────────────────

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_FastAlias_LoadsSuccessfully()
    {
        await using var model = await LocalDetector.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
        model.ClassLabels.Should().NotBeEmpty("COCO labels should be loaded");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_GetModelInfo_ReturnsValidInfo()
    {
        await using var model = await LocalDetector.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var info = model.GetModelInfo();
        info.Should().NotBeNull();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_InvalidModelId_ThrowsException()
    {
        var act = () => LocalDetector.LoadAsync("nonexistent-detector-xyz");
        await act.Should().ThrowAsync<Exception>();
    }

    // ── I axis: Basic Inference ─────────────────────────────────────

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_GradientImage_RunsWithoutError()
    {
        await using var model = await LocalDetector.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateGradientBmp(640, 480);
        var detections = await model.DetectAsync(imageBytes, TestContext.Current.CancellationToken);

        detections.Should().NotBeNull();
        // Gradient image may or may not have detections — just verify it runs
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_DetectionResults_HaveRequiredFields()
    {
        await using var model = await LocalDetector.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateGradientBmp(640, 480);
        var detections = await model.DetectAsync(imageBytes, TestContext.Current.CancellationToken);

        foreach (var d in detections)
        {
            d.Label.Should().NotBeNullOrEmpty();
            d.Confidence.Should().BeInRange(0f, 1f);
            d.Box.Width.Should().BeGreaterThanOrEqualTo(0);
            d.Box.Height.Should().BeGreaterThanOrEqualTo(0);
        }
    }

    // ── Q axis: Quality ─────────────────────────────────────────────

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_ConfidenceRange_IsZeroToOne()
    {
        await using var model = await LocalDetector.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateGradientBmp(640, 480);
        var detections = await model.DetectAsync(imageBytes, TestContext.Current.CancellationToken);

        foreach (var d in detections)
        {
            d.Confidence.Should().BeGreaterThanOrEqualTo(0f);
            d.Confidence.Should().BeLessThanOrEqualTo(1f);
        }
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_BoundingBoxes_AreWithinImageBounds()
    {
        await using var model = await LocalDetector.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        const int width = 640;
        const int height = 480;
        var imageBytes = TestDataHelper.CreateGradientBmp(width, height);
        var detections = await model.DetectAsync(imageBytes, TestContext.Current.CancellationToken);

        foreach (var d in detections)
        {
            d.Box.X1.Should().BeGreaterThanOrEqualTo(-1f, "X1 should be >= 0 (with tolerance)");
            d.Box.Y1.Should().BeGreaterThanOrEqualTo(-1f, "Y1 should be >= 0 (with tolerance)");
            d.Box.X2.Should().BeLessThanOrEqualTo(width + 1, "X2 should be <= image width");
            d.Box.Y2.Should().BeLessThanOrEqualTo(height + 1, "Y2 should be <= image height");
        }
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_ClassLabels_ContainCOCOLabels()
    {
        await using var model = await LocalDetector.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        // RT-DETR uses COCO labels
        model.ClassLabels.Should().Contain("person");
        model.ClassLabels.Should().Contain("car");
    }

    // ── E axis: Edge Cases ──────────────────────────────────────────

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_BlankWhiteImage_DoesNotCrash()
    {
        await using var model = await LocalDetector.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateSolidBmp(1920, 1080);
        var act = () => model.DetectAsync(imageBytes);
        var result = await act.Should().NotThrowAsync();
        result.Subject.Should().NotBeNull();
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_TinyImage_DoesNotCrash()
    {
        await using var model = await LocalDetector.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateTinyBmp();
        var act = () => model.DetectAsync(imageBytes);
        // May throw or return empty — either is fine, no crash
        try
        {
            var result = await act();
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
        await using var model = await LocalDetector.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var act = () => model.DetectAsync(new byte[] { 0x00, 0x01, 0x02 });
        await act.Should().ThrowAsync<Exception>("non-image data should throw, not crash");
    }

    // ── Gap Tests: Default Alias, Stream, Models, Properties ───────

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_DefaultAlias_LoadsSuccessfully()
    {
        await using var model = await LocalDetector.LoadAsync("default", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAvailableModels_ReturnsAliases()
    {
        var models = LocalDetector.GetAvailableModels().ToList();

        models.Should().NotBeEmpty();
        models.Should().Contain("fast");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_ReturnsModelInfo()
    {
        var models = LocalDetector.GetAllModels().ToList();

        models.Should().NotBeEmpty();
        models.Should().AllSatisfy(m => m.Should().NotBeNull());
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_CocoClassLabels_AreAvailable()
    {
        var labels = LocalDetector.CocoClassLabels;

        labels.Should().NotBeEmpty();
        labels.Should().Contain("person");
        labels.Should().Contain("car");
        labels.Should().HaveCountGreaterThan(10, "COCO has 80 classes");
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_DetectFromStream_Works()
    {
        await using var model = await LocalDetector.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateGradientBmp(640, 480);
        using var stream = new MemoryStream(imageBytes);
        var detections = await model.DetectAsync(stream, TestContext.Current.CancellationToken);

        detections.Should().NotBeNull();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_ModelProperties_ArePopulated()
    {
        await using var model = await LocalDetector.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        model.ActiveProviders.Should().NotBeNull();
        model.RequestedProvider.Should().BeDefined();
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_LargeImage_DoesNotCrash()
    {
        await using var model = await LocalDetector.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        // Large image (1920x1080)
        var imageBytes = TestDataHelper.CreateGradientBmp(1920, 1080);
        var act = () => model.DetectAsync(imageBytes);
        await act.Should().NotThrowAsync("large images should not crash detection");
    }

    // ── Static Tests: Model Registry Validation (no model loading) ─

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAvailableModels_ContainsAllStandardAliases()
    {
        var models = LocalDetector.GetAvailableModels().ToList();

        models.Should().Contain("default");
        models.Should().Contain("fast");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_AllHaveRequiredFields()
    {
        var models = LocalDetector.GetAllModels().ToList();

        models.Should().AllSatisfy(m =>
        {
            m.Id.Should().NotBeNullOrEmpty($"Id should be set for {m.AliasName}");
            m.AliasName.Should().NotBeNullOrEmpty($"AliasName should be set for {m.Id}");
            m.DisplayName.Should().NotBeNullOrEmpty($"DisplayName should be set for {m.AliasName}");
            m.Architecture.Should().NotBeNullOrEmpty($"Architecture should be set for {m.AliasName}");
            m.InputSize.Should().BeGreaterThan(0, $"InputSize should be positive for {m.AliasName}");
            m.NumClasses.Should().BeGreaterThan(0, $"NumClasses should be positive for {m.AliasName}");
            m.OnnxFile.Should().NotBeNullOrEmpty($"OnnxFile should be set for {m.AliasName}");
        });
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_CocoClassLabels_Has80Classes()
    {
        var labels = LocalDetector.CocoClassLabels;

        labels.Should().HaveCount(80, "COCO dataset has exactly 80 classes");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_CocoClassLabels_ContainsCommonObjects()
    {
        var labels = LocalDetector.CocoClassLabels;

        labels.Should().Contain("person");
        labels.Should().Contain("car");
        labels.Should().Contain("dog");
        labels.Should().Contain("cat");
        labels.Should().Contain("chair");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_DetectorOptions_Defaults_AreReasonable()
    {
        var opts = new DetectorOptions();

        opts.ModelId.Should().Be("default", "default model ID should be 'default'");
        opts.ConfidenceThreshold.Should().Be(0.25f, "default confidence threshold should be 0.25");
        opts.IouThreshold.Should().Be(0.45f, "default IoU threshold should be 0.45");
        opts.MaxDetections.Should().Be(100, "default max detections should be 100");
        opts.DisableAutoDownload.Should().BeFalse("auto download should be enabled by default");
        opts.ClassFilter.Should().BeNull("default class filter should be null");
        opts.Provider.Should().Be(ExecutionProvider.Auto, "default provider should be Auto");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_ThresholdInfluence_LowerThresholdMoreDetections()
    {
        // Load with low threshold
        var lowOpts = new DetectorOptions { ConfidenceThreshold = 0.01f };
        await using var lowModel = await LocalDetector.LoadAsync("fast", lowOpts, cancellationToken: TestContext.Current.CancellationToken);

        // Load with high threshold
        var highOpts = new DetectorOptions { ConfidenceThreshold = 0.99f };
        await using var highModel = await LocalDetector.LoadAsync("fast", highOpts, cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateGradientBmp(640, 480);
        var lowDetections = await lowModel.DetectAsync(imageBytes, TestContext.Current.CancellationToken);
        var highDetections = await highModel.DetectAsync(imageBytes, TestContext.Current.CancellationToken);

        lowDetections.Count.Should().BeGreaterThanOrEqualTo(highDetections.Count,
            "lower confidence threshold should produce >= detections than higher threshold");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_DetectorOptions_Clone_DeepCopiesClassFilter()
    {
        var original = new DetectorOptions
        {
            ModelId = "fast",
            ConfidenceThreshold = 0.5f,
            ClassFilter = new HashSet<int> { 0, 1, 2 }
        };

        var clone = original.Clone();

        // Verify values are copied
        clone.ModelId.Should().Be("fast");
        clone.ConfidenceThreshold.Should().Be(0.5f);
        clone.ClassFilter.Should().HaveCount(3);
        clone.ClassFilter.Should().Contain(0);
        clone.ClassFilter.Should().Contain(1);
        clone.ClassFilter.Should().Contain(2);

        // Verify ClassFilter is a deep copy (not same reference)
        ReferenceEquals(original.ClassFilter, clone.ClassFilter)
            .Should().BeFalse("clone should have a separate ClassFilter instance");
    }

    // ── High-Contrast Image Tests (D-Q1 approximation) ──────────

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_HighContrastImage_ProducesDifferentResultsThanSolid()
    {
        // D-Q1 approximation: high-contrast scene vs. blank white image
        await using var model = await LocalDetector.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var sceneImage = TestDataHelper.CreateHighContrastBmp(640, 480);
        var blankImage = TestDataHelper.CreateSolidBmp(640, 480, 255, 255, 255);

        var sceneDetections = await model.DetectAsync(sceneImage, TestContext.Current.CancellationToken);
        var blankDetections = await model.DetectAsync(blankImage, TestContext.Current.CancellationToken);

        // Blank image should have fewer or no detections
        sceneDetections.Should().NotBeNull();
        blankDetections.Should().NotBeNull();

        // High-contrast scene has distinct regions/edges — should differ from blank
        var sceneCount = sceneDetections.Count;
        var blankCount = blankDetections.Count;

        // If detector finds objects in the scene, blank should have fewer
        if (sceneCount > 0)
        {
            blankCount.Should().BeLessThanOrEqualTo(sceneCount,
                "blank image should not produce more detections than scene image");
        }
    }
}
