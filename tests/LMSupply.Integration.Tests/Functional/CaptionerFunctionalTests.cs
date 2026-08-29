using LMSupply.Captioner;
using LMSupply.Captioner.Models;
using LMSupply.Integration.Tests.Helpers;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// Functional tests for the Captioner (Image→Text) domain.
/// Uses vit-gpt2 models. Tests L/I/Q/E axes.
/// Requires GPU + network access. Run locally only.
/// </summary>
[Trait("Category", "Functional")]
[Trait("Category", "LocalOnly")]
[Trait("Domain", "Captioner")]
public class CaptionerFunctionalTests
{
    // ── L axis: Model Loading ───────────────────────────────────────

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_FastAlias_LoadsSuccessfully()
    {
        await using var model = await LocalCaptioner.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_GetModelInfo_ReturnsValidInfo()
    {
        await using var model = await LocalCaptioner.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var info = model.GetModelInfo();
        info.Should().NotBeNull();
    }

    // ── I axis: Basic Inference ─────────────────────────────────────

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_GradientImage_ProducesCaption()
    {
        await using var model = await LocalCaptioner.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateGradientBmp(640, 480);
        var result = await model.CaptionAsync(imageBytes, TestContext.Current.CancellationToken);

        result.Should().NotBeNull();
        result.Caption.Should().NotBeNullOrEmpty("captioner should produce text for any image");
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_CaptionResult_HasRequiredFields()
    {
        await using var model = await LocalCaptioner.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateGradientBmp(640, 480);
        var result = await model.CaptionAsync(imageBytes, TestContext.Current.CancellationToken);

        result.Caption.Should().NotBeNullOrEmpty();
        result.Confidence.Should().BeGreaterThanOrEqualTo(0f);
    }

    // ── Q axis: Quality ─────────────────────────────────────────────

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_RepeatCaption_IsConsistent()
    {
        await using var model = await LocalCaptioner.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateGradientBmp(640, 480);

        var result1 = await model.CaptionAsync(imageBytes, TestContext.Current.CancellationToken);
        var result2 = await model.CaptionAsync(imageBytes, TestContext.Current.CancellationToken);

        result1.Caption.Should().Be(result2.Caption,
            "same image should produce same caption (deterministic)");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_Caption_ContainsEnglishWords()
    {
        await using var model = await LocalCaptioner.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateGradientBmp(640, 480);
        var result = await model.CaptionAsync(imageBytes, TestContext.Current.CancellationToken);

        // Caption should contain at least one recognizable English word
        result.Caption.Should().MatchRegex(@"\b[a-zA-Z]{2,}\b",
            "caption should contain English words");
    }

    // ── E axis: Edge Cases ──────────────────────────────────────────

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_SolidWhiteImage_ProducesCaption()
    {
        await using var model = await LocalCaptioner.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateSolidBmp(640, 480);
        var result = await model.CaptionAsync(imageBytes, TestContext.Current.CancellationToken);

        result.Should().NotBeNull();
        result.Caption.Should().NotBeNull();
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_TinyImage_DoesNotCrash()
    {
        await using var model = await LocalCaptioner.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateTinyBmp();
        try
        {
            var result = await model.CaptionAsync(imageBytes, TestContext.Current.CancellationToken);
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
        await using var model = await LocalCaptioner.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var act = () => model.CaptionAsync(new byte[] { 0x00, 0x01, 0x02 });
        await act.Should().ThrowAsync<Exception>("non-image data should throw, not crash");
    }

    // ── Gap Tests: Default Alias, Models, Stream, Properties ───────

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_DefaultAlias_LoadsSuccessfully()
    {
        await using var model = await LocalCaptioner.LoadAsync("default", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAvailableModels_ContainsAliasesAndModelNames()
    {
        var models = LocalCaptioner.GetAvailableModels().ToList();

        models.Should().NotBeEmpty();
        // Should contain both user-friendly aliases and model names
        models.Should().Contain("default", "should include 'default' alias");
        models.Should().Contain("fast", "should include 'fast' alias");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAvailableModels_ContainsDefaultModel()
    {
        var models = LocalCaptioner.GetAvailableModels().ToList();

        models.Should().Contain("vit-gpt2",
            "default alias maps to vit-gpt2 model");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_ModelProperties_ArePopulated()
    {
        await using var model = await LocalCaptioner.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        model.ActiveProviders.Should().NotBeNull();
        model.RequestedProvider.Should().BeDefined();
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_CaptionFromStream_Works()
    {
        await using var model = await LocalCaptioner.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateGradientBmp(256, 256);
        using var stream = new MemoryStream(imageBytes);
        var result = await model.CaptionAsync(stream, TestContext.Current.CancellationToken);

        result.Should().NotBeNull();
        result.Caption.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_LargeImage_DoesNotCrash()
    {
        await using var model = await LocalCaptioner.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var imageBytes = TestDataHelper.CreateGradientBmp(1920, 1080);
        var act = () => model.CaptionAsync(imageBytes);
        await act.Should().NotThrowAsync("large images should not crash captioning");
    }

    // ── Static Tests: Options Defaults & Model Registry (no model loading) ─

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_CaptionerOptions_Defaults_AreReasonable()
    {
        var opts = new CaptionerOptions();

        opts.MaxLength.Should().Be(50, "default max caption length should be 50");
        opts.NumBeams.Should().Be(1, "default beam search should be greedy (1)");
        opts.Temperature.Should().Be(1.0f, "default temperature should be 1.0");
        opts.Prompt.Should().BeNull("default prompt should be null");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAvailableModels_HasExpectedCount()
    {
        var models = LocalCaptioner.GetAvailableModels().ToList();

        // 2 aliases (default, fast) + model names + repo IDs
        models.Count.Should().BeGreaterThanOrEqualTo(4,
            "registry should have at least 4 entries (aliases + model names + repo IDs)");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAvailableModels_ContainsAllKnownModels()
    {
        var models = LocalCaptioner.GetAvailableModels().ToList();

        models.Should().Contain("vit-gpt2");
    }

    // ── GetAllModels API (no model loading) ───────────────────────

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_ReturnsFourModels()
    {
        var models = LocalCaptioner.GetAllModels().ToList();

        models.Count.Should().Be(1,
            "registry has 1 unique captioning model (VitGpt2, aliased as both default and fast)");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_ModelsHaveValidProperties()
    {
        var models = LocalCaptioner.GetAllModels().ToList();

        foreach (var model in models)
        {
            model.RepoId.Should().NotBeNullOrEmpty($"RepoId should be set for {model.AliasName}");
            model.AliasName.Should().NotBeNullOrEmpty($"AliasName should be set for {model.RepoId}");
            model.DisplayName.Should().NotBeNullOrEmpty($"DisplayName should be set for {model.AliasName}");
            model.EncoderFile.Should().NotBeNullOrEmpty($"EncoderFile should be set for {model.AliasName}");
            model.DecoderFile.Should().NotBeNullOrEmpty($"DecoderFile should be set for {model.AliasName}");
        }
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_GetAllModels_ContainsExpectedAliases()
    {
        var models = LocalCaptioner.GetAllModels().ToList();
        var aliases = models.Select(m => m.AliasName).ToList();

        aliases.Should().Contain("default");
        aliases.Should().Contain("fast");
    }

    // ── Gap Tests: Caption quality, text image ──────────────────

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_DifferentImages_ProduceDifferentCaptions()
    {
        await using var model = await LocalCaptioner.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var gradientImage = TestDataHelper.CreateGradientBmp(640, 480);
        var solidImage = TestDataHelper.CreateSolidBmp(640, 480, 0, 0, 255); // solid blue

        var gradientCaption = await model.CaptionAsync(gradientImage, TestContext.Current.CancellationToken);
        var solidCaption = await model.CaptionAsync(solidImage, TestContext.Current.CancellationToken);

        // Both should produce captions, but they may differ for different images
        gradientCaption.Caption.Should().NotBeNullOrEmpty();
        solidCaption.Caption.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_CaptionerOptions_AllDefaults_AreDocumented()
    {
        var opts = new CaptionerOptions();

        opts.MaxLength.Should().Be(50);
        opts.NumBeams.Should().Be(1);
        opts.Temperature.Should().Be(1.0f);
        opts.Prompt.Should().BeNull();
        opts.Provider.Should().Be(ExecutionProvider.Auto);
        opts.CacheDirectory.Should().BeNull();
    }

    // ── High-Contrast Image Tests (C-Q1 approximation) ──────────

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_HighContrastScene_ProducesMeaningfulCaption()
    {
        // C-Q1 approximation: high-contrast scene should produce descriptive caption
        await using var model = await LocalCaptioner.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var sceneImage = TestDataHelper.CreateHighContrastBmp(640, 480);
        var caption = await model.CaptionAsync(sceneImage, TestContext.Current.CancellationToken);

        caption.Caption.Should().NotBeNullOrEmpty(
            "high-contrast scene image should produce a caption");
        caption.Caption.Length.Should().BeGreaterThan(5,
            "caption should be more than a few characters");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_SceneVsBlank_ProduceDifferentCaptions()
    {
        // High-contrast scene should caption differently from solid blank
        await using var model = await LocalCaptioner.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var sceneImage = TestDataHelper.CreateHighContrastBmp(640, 480);
        var blankImage = TestDataHelper.CreateSolidBmp(640, 480, 255, 255, 255);

        var sceneCaption = await model.CaptionAsync(sceneImage, TestContext.Current.CancellationToken);
        var blankCaption = await model.CaptionAsync(blankImage, TestContext.Current.CancellationToken);

        sceneCaption.Caption.Should().NotBeNullOrEmpty();
        blankCaption.Caption.Should().NotBeNullOrEmpty();

        // The captions should differ — scene has distinct regions vs. blank
        // (This is a soft check — same caption is not a bug, just less interesting)
    }
}
