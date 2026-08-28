using AwesomeAssertions;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.Tests;

public class GenerationOptionsTests
{
    [Fact]
    public void Default_ReturnsExpectedValues()
    {
        // Act
        var options = GenerationOptions.Default;

        // Assert
        options.MaxTokens.Should().Be(512);
        options.Temperature.Should().Be(0.7f);
        options.TopP.Should().Be(0.9f);
        options.TopK.Should().Be(50);
        options.RepetitionPenalty.Should().Be(1.1f);
    }

    [Fact]
    public void Creative_HasHigherTemperature()
    {
        // Act
        var options = GenerationOptions.Creative;

        // Assert
        options.Temperature.Should().Be(0.9f);
        options.TopP.Should().Be(0.95f);
        options.TopK.Should().Be(100);
    }

    [Fact]
    public void Precise_HasLowerTemperature()
    {
        // Act
        var options = GenerationOptions.Precise;

        // Assert
        options.Temperature.Should().Be(0.1f);
        options.TopP.Should().Be(0.5f);
        options.TopK.Should().Be(10);
    }

    [Fact]
    public void Default_HasExpectedSamplingOptions()
    {
        // Act
        var options = GenerationOptions.Default;

        // Assert - New options from research-05
        options.DoSample.Should().BeTrue();
        options.NumBeams.Should().Be(1);
        options.PastPresentShareBuffer.Should().BeTrue();
        options.MaxNewTokens.Should().BeNull();
    }

    [Fact]
    public void BeamSearch_Configuration()
    {
        // Arrange
        var options = new GenerationOptions
        {
            NumBeams = 4,
            DoSample = false
        };

        // Assert
        options.NumBeams.Should().Be(4);
        options.DoSample.Should().BeFalse();
    }

    [Fact]
    public void MaxNewTokens_CanBeLimited()
    {
        // Arrange
        var options = new GenerationOptions
        {
            MaxTokens = 2048,
            MaxNewTokens = 100
        };

        // Assert
        options.MaxTokens.Should().Be(2048);
        options.MaxNewTokens.Should().Be(100);
    }

    [Fact]
    public void Default_Thinking_IsAuto()
    {
        GenerationOptions.Default.Thinking.Should().Be(ThinkingMode.Auto,
            because: "the default must preserve each model's built-in thinking behavior, not force it on or off");
    }

    [Fact]
    public void Gemma4Preset_Thinking_IsAuto()
    {
        GenerationOptions.Gemma4.Thinking.Should().Be(ThinkingMode.Auto,
            because: "Gemma4 preset controls sampler params only; thinking is an independent per-call-site setting");
    }

    [Fact]
    public void Qwen3_HasExpectedSamplingParameters()
    {
        var opts = GenerationOptions.Qwen3;

        opts.Temperature.Should().BeApproximately(0.6f, 0.0001f);
        opts.TopP.Should().BeApproximately(0.95f, 0.0001f);
        opts.TopK.Should().Be(20);
        opts.MinP.Should().BeApproximately(0.0f, 0.0001f);
        opts.RepetitionPenalty.Should().BeApproximately(1.0f, 0.0001f);
    }

    // --- Length safety contract (Issue 3: ResolveMaxOutputTokens) ---
    // The shared hard cap both backends must enforce. A non-positive limit means "unset",
    // never "infinite", so generation can never be unbounded.

    [Fact]
    public void ResolveMaxOutputTokens_Default_ReturnsMaxTokens()
    {
        GenerationOptions.Default.ResolveMaxOutputTokens()
            .Should().Be(GenerationOptions.DefaultMaxTokens, because: "with no MaxNewTokens, MaxTokens (512) is the cap");
    }

    [Fact]
    public void ResolveMaxOutputTokens_MaxNewTokensSet_TakesPrecedenceOverMaxTokens()
    {
        var opts = new GenerationOptions { MaxTokens = 2048, MaxNewTokens = 100 };

        opts.ResolveMaxOutputTokens().Should().Be(100, because: "MaxNewTokens is the explicit output-only cap");
    }

    [Fact]
    public void ResolveMaxOutputTokens_MaxTokensOnly_ReturnsMaxTokens()
    {
        new GenerationOptions { MaxTokens = 2048 }.ResolveMaxOutputTokens().Should().Be(2048);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(-512)]
    public void ResolveMaxOutputTokens_NonPositiveMaxTokens_NormalizesToDefault(int maxTokens)
    {
        // Non-positive MaxTokens must NOT mean unbounded generation — it normalizes to the default cap.
        new GenerationOptions { MaxTokens = maxTokens }.ResolveMaxOutputTokens()
            .Should().Be(GenerationOptions.DefaultMaxTokens);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void ResolveMaxOutputTokens_NonPositiveMaxNewTokens_NormalizesToDefault(int maxNewTokens)
    {
        // An explicitly non-positive MaxNewTokens is invalid input, normalized rather than treated as infinite.
        new GenerationOptions { MaxTokens = 2048, MaxNewTokens = maxNewTokens }.ResolveMaxOutputTokens()
            .Should().Be(GenerationOptions.DefaultMaxTokens);
    }

    // --- Advanced anti-repetition samplers (Issue 2) ---
    // All default to null ("backend default"), so existing callers are unaffected until they opt in.

    [Fact]
    public void Default_AntiRepetitionSamplers_AreNull()
    {
        var opts = GenerationOptions.Default;

        opts.DryMultiplier.Should().BeNull();
        opts.DryBase.Should().BeNull();
        opts.DryAllowedLength.Should().BeNull();
        opts.DryPenaltyLastN.Should().BeNull();
        opts.RepeatLastN.Should().BeNull();
        opts.NoRepeatNgramSize.Should().BeNull();
    }

    [Fact]
    public void AntiRepetitionSamplers_AreAssignable()
    {
        var opts = new GenerationOptions
        {
            DryMultiplier = 0.8f,
            DryBase = 1.75f,
            DryAllowedLength = 2,
            DryPenaltyLastN = -1,
            RepeatLastN = 64,
            NoRepeatNgramSize = 3
        };

        opts.DryMultiplier.Should().BeApproximately(0.8f, 0.0001f);
        opts.DryBase.Should().BeApproximately(1.75f, 0.0001f);
        opts.DryAllowedLength.Should().Be(2);
        opts.DryPenaltyLastN.Should().Be(-1);
        opts.RepeatLastN.Should().Be(64);
        opts.NoRepeatNgramSize.Should().Be(3);
    }

    // --- AdaptiveSamplingPolicy (Issue 1): low-end-safe repetition penalty floor ---

    [Fact]
    public void AdaptiveSampling_LowEnd_RaisesDisabledPenaltyToFloor()
    {
        // The preset trap: a 1.0 (disabled) penalty on a low-end/quantized model is raised to the safe floor.
        AdaptiveSamplingPolicy.ResolveRepetitionPenalty(1.0f, isLowEnd: true)
            .Should().BeApproximately(AdaptiveSamplingPolicy.LowEndMinRepetitionPenalty, 0.0001f);
    }

    [Fact]
    public void AdaptiveSampling_LowEnd_DoesNotLowerStrongerPenalty()
    {
        // Never weakens a caller's stronger defense.
        AdaptiveSamplingPolicy.ResolveRepetitionPenalty(1.3f, isLowEnd: true)
            .Should().BeApproximately(1.3f, 0.0001f);
    }

    [Fact]
    public void AdaptiveSampling_NotLowEnd_LeavesValueUnchanged()
    {
        // High-end / full-precision: presets keep their vendor-recommended values verbatim.
        AdaptiveSamplingPolicy.ResolveRepetitionPenalty(1.0f, isLowEnd: false)
            .Should().BeApproximately(1.0f, 0.0001f);
    }

    [Fact]
    public void AdaptiveSampling_LowEnd_AtFloor_IsIdempotent()
    {
        AdaptiveSamplingPolicy.ResolveRepetitionPenalty(AdaptiveSamplingPolicy.LowEndMinRepetitionPenalty, isLowEnd: true)
            .Should().BeApproximately(AdaptiveSamplingPolicy.LowEndMinRepetitionPenalty, 0.0001f);
    }
}
