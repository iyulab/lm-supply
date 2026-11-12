using LMSupply.ImageGenerator;
using LMSupply.Integration.Tests.Helpers;
using LMSupply.Segmenter;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// Functional tests for the Segmenter (Image Segmentation) domain.
/// Uses SegFormer models. Tests L/I/Q/E axes.
/// Requires GPU + network access. Run locally only.
/// </summary>
[Trait("Category", "Functional")]
[Trait("Category", "LocalOnly")]
[Trait("Domain", "Segmenter")]
public class SegmenterFunctionalTests
{
    // ── L axis: Model Loading ───────────────────────────────────────

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_FastAlias_LoadsSuccessfully()
    {
        await using var model = await LocalSegmenter.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
        model.ClassLabels.Should().NotBeEmpty("ADE20K labels should be loaded");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_GetModelInfo_ReturnsValidInfo()
    {
        await using var model = await LocalSegmenter.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var info = model.GetModelInfo();
        info.Should().NotBeNull();
    }

    // ── I axis: Basic Inference ─────────────────────────────────────

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_BasicSegmentation_ReturnsMask()
    {
        await using var model = await LocalSegmenter.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateGradientBmp(256, 256);
        var result = await model.SegmentAsync(imageBytes, TestContext.Current.CancellationToken);

        result.Should().NotBeNull();
        result.Width.Should().BeGreaterThan(0);
        result.Height.Should().BeGreaterThan(0);
        result.ClassMap.Should().NotBeEmpty();
    }

    // ── Q axis: Quality ─────────────────────────────────────────────

    // A class map that covers every pixel and has some class says nothing about whether the classes are right: for a long time the
    // portrait model labelled every pixel background, and the query model failed on every image. These facts segment a portrait
    // (made by the image generator, whose own fact checks it follows its prompt) and ask where the person is.

    private static async Task<byte[]> MakePortraitAsync()
    {
        await using var generator = await LocalImageGenerator.LoadAsync("default", cancellationToken: TestContext.Current.CancellationToken);
        var portrait = await generator.GenerateAsync(
            "a portrait photo of a woman standing in front of a plain wall",
            new ImageGenerationOptions { Steps = 4, Seed = 42 },
            TestContext.Current.CancellationToken);
        return portrait.ImageData;
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_PortraitModel_FindsThePersonInAPortrait()
    {
        var portrait = await MakePortraitAsync();
        await using var model = await LocalSegmenter.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.SegmentAsync(portrait, TestContext.Current.CancellationToken);

        model.ClassLabels.Should().Equal("background", "person");
        var person = result.ClassMap.Count(c => c == 1) / (double)result.ClassMap.Length;
        // Observed: 30 % of the picture, where an earlier build found none.
        person.Should().BeInRange(0.1, 0.7);
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_QueryModel_NamesThePersonInAPortrait()
    {
        var portrait = await MakePortraitAsync();
        await using var model = await LocalSegmenter.LoadAsync("quality", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.SegmentAsync(portrait, TestContext.Current.CancellationToken);

        model.ClassLabels.Should().HaveCount(847);
        var person = result.ClassMap.Count(c => model.ClassLabels[c].StartsWith("person", StringComparison.Ordinal)) / (double)result.ClassMap.Length;
        // Observed: 28 % of the picture.
        person.Should().BeInRange(0.1, 0.7);
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_ClassMap_CoversAllPixels()
    {
        await using var model = await LocalSegmenter.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateGradientBmp(256, 256);
        var result = await model.SegmentAsync(imageBytes, TestContext.Current.CancellationToken);

        result.ClassMap.Length.Should().Be(result.Width * result.Height,
            "class map should have one entry per pixel");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_ClassMap_HasAtLeastOneClass()
    {
        await using var model = await LocalSegmenter.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateGradientBmp(256, 256);
        var result = await model.SegmentAsync(imageBytes, TestContext.Current.CancellationToken);

        result.UniqueClassCount.Should().BeGreaterThanOrEqualTo(1,
            "segmentation should assign at least one class");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_ConfidenceMap_HasValidValues()
    {
        await using var model = await LocalSegmenter.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateGradientBmp(256, 256);
        var result = await model.SegmentAsync(imageBytes, TestContext.Current.CancellationToken);

        if (result.ConfidenceMap.Length > 0)
        {
            result.ConfidenceMap.Should().AllSatisfy(c =>
                c.Should().BeGreaterThanOrEqualTo(0f, "confidence should be >= 0"));
        }
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_GetTopSegments_ReturnsValidSummaries()
    {
        await using var model = await LocalSegmenter.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateGradientBmp(256, 256);
        var result = await model.SegmentAsync(imageBytes, TestContext.Current.CancellationToken);

        var topSegments = result.GetTopSegments(5, model.ClassLabels);

        topSegments.Should().NotBeEmpty();
        foreach (var seg in topSegments)
        {
            seg.PixelCount.Should().BeGreaterThan(0);
            seg.CoverageRatio.Should().BeInRange(0f, 1f);
            seg.Label.Should().NotBeNullOrEmpty();
        }
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_ClassLabels_ContainADE20KLabels()
    {
        // The default model is a SegFormer, which is trained on the 150 ADE20K classes (the portrait model `fast` has two).
        await using var model = await LocalSegmenter.LoadAsync("default", cancellationToken: TestContext.Current.CancellationToken);

        model.ClassLabels.Should().HaveCount(150);
        model.ClassLabels.Should().Contain("wall");
    }

    // ── E axis: Edge Cases ──────────────────────────────────────────

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_SolidWhiteImage_ProducesSingleClass()
    {
        await using var model = await LocalSegmenter.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateSolidBmp(256, 256);
        var act = () => model.SegmentAsync(imageBytes);
        var result = await act.Should().NotThrowAsync();
        result.Subject.ClassMap.Should().NotBeEmpty();
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_TinyImage_DoesNotCrash()
    {
        await using var model = await LocalSegmenter.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateTinyBmp();
        try
        {
            var result = await model.SegmentAsync(imageBytes, TestContext.Current.CancellationToken);
            result.Should().NotBeNull();
        }
        catch (Exception ex)
        {
            ex.Should().NotBeOfType<NullReferenceException>();
        }
    }

    // ── Gap Tests: Default Alias, Models, Stream, Properties ───────

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_DefaultAlias_LoadsSuccessfully()
    {
        await using var model = await LocalSegmenter.LoadAsync("default", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAvailableModels_ReturnsAliases()
    {
        var models = LocalSegmenter.GetAvailableModels().ToList();

        models.Should().NotBeEmpty();
        models.Should().Contain("fast");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_ReturnsModelInfo()
    {
        var models = LocalSegmenter.GetAllModels().ToList();

        models.Should().NotBeEmpty();
        models.Should().AllSatisfy(m => m.Should().NotBeNull());
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_ModelProperties_ArePopulated()
    {
        await using var model = await LocalSegmenter.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        model.ActiveProviders.Should().NotBeNull();
        model.RequestedProvider.Should().BeDefined();
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_SegmentFromStream_Works()
    {
        await using var model = await LocalSegmenter.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateGradientBmp(256, 256);
        using var stream = new MemoryStream(imageBytes);
        var result = await model.SegmentAsync(stream, TestContext.Current.CancellationToken);

        result.Should().NotBeNull();
        result.ClassMap.Should().NotBeEmpty();
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_LargeImage_DoesNotCrash()
    {
        await using var model = await LocalSegmenter.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateGradientBmp(1920, 1080);
        var act = () => model.SegmentAsync(imageBytes);
        await act.Should().NotThrowAsync("large images should not crash segmentation");
    }

    // ── Static Tests: Model Registry Validation (no model loading) ─

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAvailableModels_ContainsAllStandardAliases()
    {
        var models = LocalSegmenter.GetAvailableModels().ToList();

        models.Should().Contain("default");
        models.Should().Contain("fast");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_AllHaveRequiredFields()
    {
        var models = LocalSegmenter.GetAllModels().ToList();

        models.Should().AllSatisfy(m =>
        {
            m.Id.Should().NotBeNullOrEmpty($"Id should be set for {m.AliasName}");
            m.AliasName.Should().NotBeNullOrEmpty($"AliasName should be set for {m.Id}");
            m.DisplayName.Should().NotBeNullOrEmpty($"DisplayName should be set for {m.AliasName}");
            m.Architecture.Should().NotBeNullOrEmpty($"Architecture should be set for {m.AliasName}");
            m.InputSize.Should().BeGreaterThan(0, $"InputSize should be positive for {m.AliasName}");
            m.OnnxFile.Should().NotBeNullOrEmpty($"OnnxFile should be set for {m.AliasName}");
        });
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_HasExpectedCount()
    {
        var models = LocalSegmenter.GetAllModels().ToList();

        models.Count.Should().BeGreaterThanOrEqualTo(2,
            "registry should have at least 2 segmentation models");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_SegmenterOptions_Defaults_AreReasonable()
    {
        var opts = new SegmenterOptions();

        opts.ModelId.Should().Be("default", "default model ID should be 'default'");
        opts.DisableAutoDownload.Should().BeFalse("auto download should be enabled by default");
        opts.ResizeToOriginal.Should().BeTrue("output should resize to original dimensions by default");
        opts.Provider.Should().Be(ExecutionProvider.Auto, "default provider should be Auto");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_MaskResolution_MatchesInputWhenResizeToOriginal()
    {
        var opts = new SegmenterOptions { ResizeToOriginal = true };
        await using var model = await LocalSegmenter.LoadAsync("fast", opts, cancellationToken: TestContext.Current.CancellationToken);

        const int width = 400;
        const int height = 300;
        var imageBytes = TestDataHelper.CreateGradientBmp(width, height);
        var result = await model.SegmentAsync(imageBytes, TestContext.Current.CancellationToken);

        result.Width.Should().Be(width, "mask width should match input when ResizeToOriginal=true");
        result.Height.Should().Be(height, "mask height should match input when ResizeToOriginal=true");
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_PanoramaImage_DoesNotCrash()
    {
        await using var model = await LocalSegmenter.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        // Very wide panorama: 2000x200
        var imageBytes = TestDataHelper.CreateGradientBmp(2000, 200);
        var act = () => model.SegmentAsync(imageBytes);
        var result = await act.Should().NotThrowAsync("panorama aspect ratio should not crash");
        result.Subject.ClassMap.Should().NotBeEmpty();
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_SegmenterOptions_Clone_IsIndependentCopy()
    {
        var original = new SegmenterOptions
        {
            ModelId = "quality",
            ResizeToOriginal = false,
            Provider = ExecutionProvider.Cpu
        };

        var clone = original.Clone();

        clone.ModelId.Should().Be("quality");
        clone.ResizeToOriginal.Should().BeFalse();

        clone.ModelId = "fast";
        original.ModelId.Should().Be("quality", "original unaffected by clone mutation");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_Ade20kClassLabels_Has150Classes()
    {
        var labels = LocalSegmenter.Ade20kClassLabels;

        labels.Should().HaveCount(150, "ADE20K dataset has 150 classes");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_Ade20kClassLabels_ContainsCommonLabels()
    {
        var labels = LocalSegmenter.Ade20kClassLabels;

        labels.Should().Contain("wall", "ADE20K should include 'wall'");
        labels.Should().Contain("floor", "ADE20K should include 'floor'");
        labels.Should().Contain("ceiling", "ADE20K should include 'ceiling'");
        labels.Should().Contain("person", "ADE20K should include 'person'");
        labels.Should().Contain("tree", "ADE20K should include 'tree'");
    }
}
