using LMSupply.ImageGenerator;
using LMSupply.ImageGenerator.Models;
using LMSupply.ImageGenerator.Schedulers;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// Functional tests for the ImageGenerator (Text→Image) domain.
/// Uses LCM-Dreamshaper-V7 models. Tests L/I/Q/E axes.
/// Requires GPU + network access. Run locally only.
/// </summary>
[Trait("Category", "Functional")]
[Trait("Category", "LocalOnly")]
[Trait("Domain", "ImageGenerator")]
public class ImageGeneratorFunctionalTests
{
    // ── L axis: Model Loading ───────────────────────────────────────

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_DefaultAlias_LoadsSuccessfully()
    {
        await using var model = await LocalImageGenerator.LoadAsync("default", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_FastAlias_LoadsSuccessfully()
    {
        await using var model = await LocalImageGenerator.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        model.ModelId.Should().NotBeNullOrEmpty();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_GetModelInfo_ReturnsValidInfo()
    {
        await using var model = await LocalImageGenerator.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var info = model.GetModelInfo();
        info.Should().NotBeNull();
        info!.ModelId.Should().NotBeNullOrEmpty();
        info.Architecture.Should().NotBeNullOrEmpty();
        info.RecommendedSteps.Should().BeGreaterThan(0);
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_GetAvailableAliases_ReturnsKnownAliases()
    {
        var aliases = LocalImageGenerator.GetAvailableAliases();

        aliases.Should().NotBeEmpty();
        aliases.Should().Contain("default");
        aliases.Should().Contain("fast");
        aliases.Should().Contain("quality");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_InvalidModelId_ThrowsArgumentException()
    {
        var act = async () => await LocalImageGenerator.LoadAsync("");

        await act.Should().ThrowAsync<ArgumentException>();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_NonExistentLocalPath_ThrowsDirectoryNotFoundException()
    {
        var act = async () => await LocalImageGenerator.LoadAsync("./nonexistent_dir_12345");

        await act.Should().ThrowAsync<DirectoryNotFoundException>();
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_WarmupAsync_CompletesWithoutError()
    {
        await using var model = await LocalImageGenerator.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        await model.WarmupAsync(TestContext.Current.CancellationToken);
        // No exception = success
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public async Task L_EstimatedMemoryBytes_IsPositive()
    {
        await using var model = await LocalImageGenerator.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        model.EstimatedMemoryBytes.Should().NotBeNull();
        model.EstimatedMemoryBytes!.Value.Should().BeGreaterThan(0);
    }

    // ── I axis: Basic Inference ─────────────────────────────────────

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_BasicGeneration_ReturnsImage()
    {
        await using var model = await LocalImageGenerator.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.GenerateAsync("A red circle on white background", new GenerationOptions { Steps = 2, Seed = 42 }, TestContext.Current.CancellationToken);

        result.Should().NotBeNull();
        result.ImageData.Should().NotBeEmpty();
        result.Width.Should().Be(512);
        result.Height.Should().Be(512);
        result.Steps.Should().Be(2);
        result.Prompt.Should().Be("A red circle on white background");
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_GenerationTime_IsPositive()
    {
        await using var model = await LocalImageGenerator.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.GenerateAsync("sunset", new GenerationOptions { Steps = 2, Seed = 42 }, TestContext.Current.CancellationToken);

        result.GenerationTime.Should().BeGreaterThan(TimeSpan.Zero);
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_BatchGeneration_ReturnsMultipleImages()
    {
        await using var model = await LocalImageGenerator.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var results = await model.GenerateBatchAsync("mountains", count: 2, new GenerationOptions { Steps = 2, Seed = 100 }, cancellationToken: TestContext.Current.CancellationToken);

        results.Should().HaveCount(2);
        results[0].ImageData.Should().NotBeEmpty();
        results[1].ImageData.Should().NotBeEmpty();
        // Batch uses seed increment: 100, 101
        results[0].Seed.Should().Be(100);
        results[1].Seed.Should().Be(101);
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_StreamingGeneration_YieldsSteps()
    {
        await using var model = await LocalImageGenerator.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var steps = new List<GenerationStep>();
        await foreach (var step in model.GenerateStreamingAsync("forest", new GenerationOptions { Steps = 2, Seed = 42 }, TestContext.Current.CancellationToken))
        {
            steps.Add(step);
        }

        steps.Should().HaveCount(2);
        steps[0].StepNumber.Should().Be(1);
        steps[0].TotalSteps.Should().Be(2);
        steps[0].IsFinal.Should().BeFalse();
        steps[1].StepNumber.Should().Be(2);
        steps[1].IsFinal.Should().BeTrue();
        steps[1].FinalImage.Should().NotBeNull();
        steps[1].FinalImage!.ImageData.Should().NotBeEmpty();
    }

    [Fact]
    [Trait("Axis", "Inference")]
    public async Task I_NegativePrompt_ExecutesWithoutError()
    {
        await using var model = await LocalImageGenerator.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.GenerateAsync("A beautiful landscape", new GenerationOptions
            {
                Steps = 2,
                Seed = 42,
                GuidanceScale = 1.5f,
                NegativePrompt = "blurry, low quality"
            }, TestContext.Current.CancellationToken);

        result.Should().NotBeNull();
        result.ImageData.Should().NotBeEmpty();
    }

    // ── Q axis: Quality Assertions ──────────────────────────────────

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_OutputIsPng_HasPngSignature()
    {
        await using var model = await LocalImageGenerator.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.GenerateAsync("test image", new GenerationOptions { Steps = 2, Seed = 42 }, TestContext.Current.CancellationToken);

        // PNG signature: 137 80 78 71 13 10 26 10
        result.ImageData.Length.Should().BeGreaterThan(8);
        result.ImageData[0].Should().Be(0x89);
        result.ImageData[1].Should().Be(0x50); // P
        result.ImageData[2].Should().Be(0x4E); // N
        result.ImageData[3].Should().Be(0x47); // G
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_SeedReproducibility_SameResultsForSameSeed()
    {
        await using var model = await LocalImageGenerator.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var options = new GenerationOptions { Steps = 2, Seed = 42 };
        var result1 = await model.GenerateAsync("cat", options, TestContext.Current.CancellationToken);
        var result2 = await model.GenerateAsync("cat", options, TestContext.Current.CancellationToken);

        result1.Seed.Should().Be(42);
        result2.Seed.Should().Be(42);
        result1.ImageData.Should().BeEquivalentTo(result2.ImageData,
            "Same seed should produce identical images");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_CustomDimensions_MatchesRequestedSize()
    {
        await using var model = await LocalImageGenerator.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.GenerateAsync("test", new GenerationOptions
            {
                Width = 256,
                Height = 256,
                Steps = 2,
                Seed = 42
            }, TestContext.Current.CancellationToken);

        result.Width.Should().Be(256);
        result.Height.Should().Be(256);
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_StreamingProgress_IsMonotonic()
    {
        await using var model = await LocalImageGenerator.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var progressValues = new List<float>();
        await foreach (var step in model.GenerateStreamingAsync("sunset", new GenerationOptions { Steps = 4, Seed = 42 }, TestContext.Current.CancellationToken))
        {
            progressValues.Add(step.Progress);
        }

        progressValues.Should().BeInAscendingOrder("progress should monotonically increase");
        progressValues.Last().Should().Be(100f);
    }

    // ── E axis: Edge Cases ──────────────────────────────────────────

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_InvalidWidth_ThrowsArgumentException()
    {
        await using var model = await LocalImageGenerator.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var act = async () => await model.GenerateAsync("test",
            new GenerationOptions { Width = 511, Steps = 2 });

        await act.Should().ThrowAsync<ArgumentException>()
            .WithMessage("*Width*divisible by 8*");
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_InvalidHeight_ThrowsArgumentException()
    {
        await using var model = await LocalImageGenerator.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var act = async () => await model.GenerateAsync("test",
            new GenerationOptions { Height = 513, Steps = 2 });

        await act.Should().ThrowAsync<ArgumentException>()
            .WithMessage("*Height*divisible by 8*");
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_ZeroSteps_ThrowsArgumentException()
    {
        await using var model = await LocalImageGenerator.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var act = async () => await model.GenerateAsync("test",
            new GenerationOptions { Steps = 0 });

        await act.Should().ThrowAsync<ArgumentException>()
            .WithMessage("*Steps*between 1 and 50*");
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_NegativeGuidanceScale_ThrowsArgumentException()
    {
        await using var model = await LocalImageGenerator.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var act = async () => await model.GenerateAsync("test",
            new GenerationOptions { GuidanceScale = -1.0f, Steps = 2 });

        await act.Should().ThrowAsync<ArgumentException>()
            .WithMessage("*GuidanceScale*non-negative*");
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_EmptyPrompt_ThrowsOrReturnsResult()
    {
        await using var model = await LocalImageGenerator.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        // Empty prompt may throw ArgumentException or produce a result
        // depending on model behavior — should NOT crash
        try
        {
            var result = await model.GenerateAsync("", new GenerationOptions { Steps = 2, Seed = 42 }, TestContext.Current.CancellationToken);
            result.Should().NotBeNull();
        }
        catch (ArgumentException)
        {
            // Acceptable: model rejects empty prompt
        }
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_SpecialCharactersInPrompt_DoesNotCrash()
    {
        await using var model = await LocalImageGenerator.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.GenerateAsync("🎨 Abstract art with 日本語 text!", new GenerationOptions { Steps = 2, Seed = 42 }, TestContext.Current.CancellationToken);

        result.Should().NotBeNull();
        result.ImageData.Should().NotBeEmpty();
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_LongPrompt_DoesNotCrash()
    {
        await using var model = await LocalImageGenerator.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var longPrompt = string.Join(", ", Enumerable.Repeat("beautiful mountain landscape", 50));

        var result = await model.GenerateAsync(longPrompt, new GenerationOptions { Steps = 2, Seed = 42 }, TestContext.Current.CancellationToken);

        result.Should().NotBeNull();
        result.ImageData.Should().NotBeEmpty();
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_NullOptions_UsesDefaults()
    {
        await using var model = await LocalImageGenerator.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.GenerateAsync("test", options: null, cancellationToken: TestContext.Current.CancellationToken);

        result.Should().NotBeNull();
        result.ImageData.Should().NotBeEmpty();
        result.Width.Should().Be(512);
        result.Height.Should().Be(512);
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_SaveAsync_WritesToFile()
    {
        await using var model = await LocalImageGenerator.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.GenerateAsync("test", new GenerationOptions { Steps = 2, Seed = 42 }, TestContext.Current.CancellationToken);

        var tempPath = Path.Combine(Path.GetTempPath(), $"lmsupply_test_{Guid.NewGuid()}.png");
        try
        {
            await result.SaveAsync(tempPath, TestContext.Current.CancellationToken);
            File.Exists(tempPath).Should().BeTrue();
            var savedBytes = await File.ReadAllBytesAsync(tempPath, TestContext.Current.CancellationToken);
            savedBytes.Should().BeEquivalentTo(result.ImageData);
        }
        finally
        {
            File.Delete(tempPath);
        }
    }

    [Fact]
    [Trait("Axis", "EdgeCase")]
    public async Task E_ToStream_ReturnsReadableStream()
    {
        await using var model = await LocalImageGenerator.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.GenerateAsync("test", new GenerationOptions { Steps = 2, Seed = 42 }, TestContext.Current.CancellationToken);

        using var stream = result.ToStream();
        stream.CanRead.Should().BeTrue();
        stream.Length.Should().Be(result.ImageData.Length);
    }

    // ── R axis: Resource Management ─────────────────────────────────

    [Fact]
    [Trait("Axis", "Resource")]
    public async Task R_DoubleDispose_NoException()
    {
        var model = await LocalImageGenerator.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        await model.DisposeAsync();
        await model.DisposeAsync();
        // No exception = success
    }

    // ── Static Tests: Options Defaults (no model loading) ────────

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_GenerationOptions_Defaults_AreReasonable()
    {
        var opts = new GenerationOptions();

        opts.Steps.Should().BeGreaterThan(0, "default steps should be positive");
        opts.Width.Should().Be(512, "default width should be 512");
        opts.Height.Should().Be(512, "default height should be 512");
        opts.GuidanceScale.Should().BeGreaterThan(0, "guidance scale should be positive");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_GenerationOptions_AllDefaults_AreDocumented()
    {
        var opts = new GenerationOptions();

        opts.NegativePrompt.Should().BeNull("default negative prompt should be null");
        opts.Width.Should().Be(512, "default width should be 512");
        opts.Height.Should().Be(512, "default height should be 512");
        opts.Steps.Should().Be(4, "default steps should be 4 (LCM)");
        opts.GuidanceScale.Should().Be(1.0f, "default guidance scale should be 1.0 (LCM)");
        opts.Seed.Should().BeNull("default seed should be null (random)");
        opts.GeneratePreviews.Should().BeFalse("default should not generate previews");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_WellKnownImageModels_HasExpectedAliases()
    {
        var aliases = ImageGeneratorModelRegistry.Default.GetAliases().Select(a => a.Name).ToList();

        aliases.Should().Contain("default");
        aliases.Should().Contain("fast");
        aliases.Should().Contain("quality");
        aliases.Should().Contain("lcm-dreamshaper-v7");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_WellKnownImageModels_IsAlias_WorksCorrectly()
    {
        var aliasNames = ImageGeneratorModelRegistry.Default.GetAliases().Select(a => a.Name).ToHashSet(StringComparer.OrdinalIgnoreCase);
        aliasNames.Should().Contain("default");
        aliasNames.Should().Contain("fast");
        aliasNames.Should().Contain("quality");
        aliasNames.Should().NotContain("nonexistent");
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_WellKnownImageModels_Resolve_ReturnsValidDefinition()
    {
        var info = ImageGeneratorModelRegistry.Default.Resolve("default");

        info.ModelId.Should().NotBeNullOrEmpty();
        info.ModelName.Should().NotBeNullOrEmpty();
        info.RecommendedSteps.Should().BeGreaterThan(0);
        info.RecommendedGuidanceScale.Should().BeGreaterThan(0);
    }

    [Fact]
    [Trait("Axis", "Loading")]
    public void L_WellKnownImageModels_FastHasFewerSteps()
    {
        var fast = ImageGeneratorModelRegistry.Default.Resolve("fast");
        var quality = ImageGeneratorModelRegistry.Default.Resolve("quality");

        fast.RecommendedSteps.Should().BeLessThan(quality.RecommendedSteps,
            "fast should use fewer steps than quality");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_BetaSchedule_HasExpectedValues()
    {
        var values = Enum.GetValues<BetaSchedule>();
        values.Should().HaveCount(3);
        values.Should().Contain(BetaSchedule.Linear);
        values.Should().Contain(BetaSchedule.ScaledLinear);
        values.Should().Contain(BetaSchedule.SquaredCosCapV2);
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_PredictionType_HasExpectedValues()
    {
        var values = Enum.GetValues<PredictionType>();
        values.Should().HaveCount(3);
        values.Should().Contain(PredictionType.Epsilon);
        values.Should().Contain(PredictionType.Sample);
        values.Should().Contain(PredictionType.VPrediction);
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public async Task Q_DifferentPrompts_ProduceDifferentImages()
    {
        await using var model = await LocalImageGenerator.LoadAsync("fast", cancellationToken: TestContext.Current.CancellationToken);

        var options = new GenerationOptions { Steps = 2, Seed = 42 };
        var result1 = await model.GenerateAsync("a red ball on a green field", options, TestContext.Current.CancellationToken);
        var result2 = await model.GenerateAsync("a blue fish in the ocean", options, TestContext.Current.CancellationToken);

        result1.ImageData.Should().NotBeEmpty();
        result2.ImageData.Should().NotBeEmpty();

        // Different prompts with same seed should produce different images
        result1.ImageData.Should().NotBeEquivalentTo(result2.ImageData,
            "different prompts should produce different images even with same seed");
    }

    [Fact]
    [Trait("Axis", "Quality")]
    public void Q_LcmSchedulerConfig_ForLcm_HasValidDefaults()
    {
        var config = LcmSchedulerConfig.ForLcm();

        config.NumTrainTimesteps.Should().Be(1000);
        config.BetaStart.Should().BeApproximately(0.00085f, 0.00001f);
        config.BetaEnd.Should().BeApproximately(0.012f, 0.001f);
        config.BetaSchedule.Should().Be(BetaSchedule.ScaledLinear);
        config.OriginalInferenceSteps.Should().Be(50);
        config.SetAlphaToOne.Should().BeTrue();
        config.PredictionType.Should().Be(PredictionType.Epsilon);
    }
}
