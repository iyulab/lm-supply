using AwesomeAssertions;
using LMSupply.Transcriber.Internal;
using LMSupply.Transcriber.Models;

namespace LMSupply.Transcriber.Tests;

public class TranscriberModelInfoTests
{
    private static TranscriberModelInfo CreateFallbackInfo() => new()
    {
        Id = "org/some-whisper-model",
        AliasName = "org/some-whisper-model",
        DisplayName = "some-whisper-model",
        Architecture = "Whisper",
        // Defaults: NumMelBins=80, HiddenSize=512
    };

    [Fact]
    public void WithConfigOverrides_BothFields_ShouldOverride()
    {
        var original = CreateFallbackInfo();
        var config = new WhisperModelConfig { NumMelBins = 128, HiddenSize = 1280 };

        var updated = original.WithConfigOverrides(config);

        updated.NumMelBins.Should().Be(128);
        updated.HiddenSize.Should().Be(1280);
    }

    [Fact]
    public void WithConfigOverrides_OnlyNumMelBins_ShouldOverridePartially()
    {
        var original = CreateFallbackInfo();
        var config = new WhisperModelConfig { NumMelBins = 128, HiddenSize = null };

        var updated = original.WithConfigOverrides(config);

        updated.NumMelBins.Should().Be(128);
        updated.HiddenSize.Should().Be(512, "original default should be preserved");
    }

    [Fact]
    public void WithConfigOverrides_OnlyHiddenSize_ShouldOverridePartially()
    {
        var original = CreateFallbackInfo();
        var config = new WhisperModelConfig { NumMelBins = null, HiddenSize = 768 };

        var updated = original.WithConfigOverrides(config);

        updated.NumMelBins.Should().Be(80, "original default should be preserved");
        updated.HiddenSize.Should().Be(768);
    }

    [Fact]
    public void WithConfigOverrides_ShouldPreserveAllOtherFields()
    {
        var original = new TranscriberModelInfo
        {
            Id = "org/test-model",
            AliasName = "org/test-model",
            DisplayName = "Test Model",
            Architecture = "Whisper",
            ParametersM = 809f,
            SizeBytes = 3_200_000_000,
            WerLibriSpeech = 2.7f,
            MaxDurationSeconds = 30,
            SampleRate = 16000,
            NumMelBins = 80,
            HiddenSize = 512,
            EncoderFile = "encoder_model.onnx",
            DecoderFile = "decoder_model_merged.onnx",
            IsMultilingual = true,
            Description = "Test description",
            License = "MIT"
        };
        var config = new WhisperModelConfig { NumMelBins = 128, HiddenSize = 1280 };

        var updated = original.WithConfigOverrides(config);

        updated.Id.Should().Be("org/test-model");
        updated.AliasName.Should().Be("org/test-model");
        updated.DisplayName.Should().Be("Test Model");
        updated.Architecture.Should().Be("Whisper");
        updated.ParametersM.Should().Be(809f);
        updated.SizeBytes.Should().Be(3_200_000_000);
        updated.WerLibriSpeech.Should().Be(2.7f);
        updated.MaxDurationSeconds.Should().Be(30);
        updated.SampleRate.Should().Be(16000);
        updated.EncoderFile.Should().Be("encoder_model.onnx");
        updated.DecoderFile.Should().Be("decoder_model_merged.onnx");
        updated.IsMultilingual.Should().BeTrue();
        updated.Description.Should().Be("Test description");
        updated.License.Should().Be("MIT");
    }

    [Fact]
    public void WithConfigOverrides_ShouldNotMutateOriginal()
    {
        var original = CreateFallbackInfo();
        var config = new WhisperModelConfig { NumMelBins = 128, HiddenSize = 1280 };

        _ = original.WithConfigOverrides(config);

        original.NumMelBins.Should().Be(80, "original instance should not be modified");
        original.HiddenSize.Should().Be(512, "original instance should not be modified");
    }

    [Fact]
    public void IsTranslateSupported_MultilingualModel_ShouldBeTrue()
    {
        DefaultModels.WhisperBase.IsMultilingual.Should().BeTrue();
        DefaultModels.WhisperBase.IsTranslateSupported.Should().BeTrue();
        DefaultModels.WhisperLargeV3Turbo.IsTranslateSupported.Should().BeTrue();
    }

    [Fact]
    public void IsTranslateSupported_EnglishOnlyModel_ShouldBeFalse()
    {
        DefaultModels.WhisperBaseEn.IsMultilingual.Should().BeFalse();
        DefaultModels.WhisperBaseEn.IsTranslateSupported.Should().BeFalse();
        DefaultModels.DistilWhisperLargeV3.IsTranslateSupported.Should().BeFalse();
    }

    [Fact]
    public void IsTranslateSupported_ShouldTrackIsMultilingual()
    {
        foreach (var model in DefaultModels.All)
        {
            model.IsTranslateSupported.Should().Be(
                model.IsMultilingual,
                $"{model.Id} translate capability should mirror IsMultilingual");
        }
    }
}
