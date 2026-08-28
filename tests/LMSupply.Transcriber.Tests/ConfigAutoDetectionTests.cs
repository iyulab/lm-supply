using AwesomeAssertions;
using LMSupply.Transcriber.Internal;
using LMSupply.Transcriber.Models;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// Tests for the config.json auto-detection pipeline integration.
/// Verifies that fallback models can be correctly updated with config.json values.
/// </summary>
public class ConfigAutoDetectionTests
{
    [Fact]
    public void FallbackModel_HasAliasNameEqualToId()
    {
        // The pipeline identifies fallback models by AliasName == Id
        var registry = new TranscriberModelRegistry(DefaultModels.All);
        var model = registry.Resolve("custom-org/whisper-large-v3-custom-finetuned");

        model.AliasName.Should().Be(model.Id, "fallback models should have AliasName == Id");
    }

    [Fact]
    public void SystemModel_HasAliasNameDifferentFromId()
    {
        // System models should NOT be overridden (AliasName != Id)
        var model = TranscriberModelRegistry.Default.Resolve("default");

        model.AliasName.Should().NotBe(model.Id, "system models have distinct alias names");
        model.AliasName.Should().Be("default");
    }

    [Fact]
    public void FallbackModel_DefaultValues_AreIncorrectForLargeV3()
    {
        // This demonstrates the problem the auto-detection solves
        var registry = new TranscriberModelRegistry(DefaultModels.All);
        var model = registry.Resolve("custom-org/whisper-large-v3-custom-finetuned");

        // Default fallback values are wrong for large-v3 models
        model.NumMelBins.Should().Be(80, "fallback defaults to 80 (wrong for large-v3)");
        model.HiddenSize.Should().Be(512, "fallback defaults to 512 (wrong for large-v3)");
    }

    [Fact]
    public void FallbackModel_WithConfigOverrides_FixesLargeV3Parameters()
    {
        // Simulate what the load pipeline does after downloading
        var registry = new TranscriberModelRegistry(DefaultModels.All);
        var model = registry.Resolve("custom-org/whisper-large-v3-custom-finetuned");

        // Simulate reading config.json
        var configJson = """
        {
            "model_type": "whisper",
            "num_mel_bins": 128,
            "d_model": 1280
        }
        """;
        var config = WhisperConfigReader.ParseConfig(configJson);

        // Apply overrides (as the load pipeline would do)
        var updated = model.WithConfigOverrides(config!);

        updated.NumMelBins.Should().Be(128, "should use value from config.json");
        updated.HiddenSize.Should().Be(1280, "should use value from config.json");
        updated.Id.Should().Be("custom-org/whisper-large-v3-custom-finetuned");
        updated.Architecture.Should().Be("Whisper");
    }

    [Fact]
    public void FallbackModel_WithConfigOverrides_TinyModel_PreservesDefaults()
    {
        // For tiny models, config values match the defaults
        var registry = new TranscriberModelRegistry(DefaultModels.All);
        var model = registry.Resolve("some-org/whisper-tiny-custom");

        var configJson = """
        {
            "model_type": "whisper",
            "num_mel_bins": 80,
            "d_model": 384
        }
        """;
        var config = WhisperConfigReader.ParseConfig(configJson);
        var updated = model.WithConfigOverrides(config!);

        updated.NumMelBins.Should().Be(80);
        updated.HiddenSize.Should().Be(384);
    }

    [Fact]
    public void FallbackModel_NoConfigJson_ShouldKeepDefaults()
    {
        // When config.json doesn't exist, fallback defaults should remain
        var config = WhisperConfigReader.ReadConfig(Path.Combine(Path.GetTempPath(), $"nonexistent-{Guid.NewGuid():N}"));

        config.Should().BeNull();
        // In this case, the pipeline skips the override and uses fallback defaults
    }

    [Fact]
    public void Pipeline_OnlyOverridesFallbackModels()
    {
        // Verify that the AliasName == Id check correctly identifies fallback vs system models
        var registry = new TranscriberModelRegistry(DefaultModels.All);

        // System model: AliasName != Id
        var systemModel = registry.Resolve("default");
        var isFallback = systemModel.AliasName == systemModel.Id;
        isFallback.Should().BeFalse("system models should not be treated as fallback");

        // Fallback model: AliasName == Id
        var fallbackModel = registry.Resolve("org/unknown-model");
        var isFallback2 = fallbackModel.AliasName == fallbackModel.Id;
        isFallback2.Should().BeTrue("fallback models should be identified by AliasName == Id");
    }

    [Fact]
    public void Pipeline_SmallModel_ConfigOverrides()
    {
        // Whisper Small: 80 mel bins, d_model=768
        var registry = new TranscriberModelRegistry(DefaultModels.All);
        var model = registry.Resolve("custom-org/whisper-small-finetuned");

        var configJson = """
        {
            "model_type": "whisper",
            "num_mel_bins": 80,
            "d_model": 768
        }
        """;
        var config = WhisperConfigReader.ParseConfig(configJson);
        var updated = model.WithConfigOverrides(config!);

        updated.NumMelBins.Should().Be(80);
        updated.HiddenSize.Should().Be(768);
    }

    [Fact]
    public void Pipeline_MediumModel_ConfigOverrides()
    {
        // Whisper Medium: 80 mel bins, d_model=1024
        var registry = new TranscriberModelRegistry(DefaultModels.All);
        var model = registry.Resolve("custom-org/whisper-medium-finetuned");

        var configJson = """
        {
            "model_type": "whisper",
            "num_mel_bins": 80,
            "d_model": 1024
        }
        """;
        var config = WhisperConfigReader.ParseConfig(configJson);
        var updated = model.WithConfigOverrides(config!);

        updated.NumMelBins.Should().Be(80);
        updated.HiddenSize.Should().Be(1024);
    }
}
