using FluentAssertions;

namespace LMSupply.Translator.Tests.Decoding;

public class BeamSearchDecoderTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(8)]
    public void BeamWidth_ShouldBeConfigurable(int beamWidth)
    {
        var options = new TranslatorOptions { BeamWidth = beamWidth };

        options.BeamWidth.Should().Be(beamWidth);
    }

    [Fact]
    public void DefaultOptions_ShouldHaveBeamWidth4()
    {
        var options = new TranslatorOptions();

        options.BeamWidth.Should().Be(4);
    }

    [Fact]
    public void DefaultOptions_ShouldUseBeamSearch()
    {
        var options = new TranslatorOptions();

        options.UseGreedyDecoding.Should().BeFalse();
    }

    [Fact]
    public void GreedyDecoding_CanBeEnabled()
    {
        var options = new TranslatorOptions { UseGreedyDecoding = true };

        options.UseGreedyDecoding.Should().BeTrue();
    }

    [Theory]
    [InlineData(0.5f)]
    [InlineData(1.0f)]
    [InlineData(1.5f)]
    public void LengthPenalty_ShouldBeConfigurable(float penalty)
    {
        var options = new TranslatorOptions { LengthPenalty = penalty };

        options.LengthPenalty.Should().Be(penalty);
    }

    [Theory]
    [InlineData(1.0f)]
    [InlineData(1.2f)]
    [InlineData(1.5f)]
    public void RepetitionPenalty_ShouldBeConfigurable(float penalty)
    {
        var options = new TranslatorOptions { RepetitionPenalty = penalty };

        options.RepetitionPenalty.Should().Be(penalty);
    }

    [Fact]
    public void DefaultOptions_ShouldHaveLengthPenalty1()
    {
        var options = new TranslatorOptions();

        options.LengthPenalty.Should().Be(1.0f);
    }

    [Fact]
    public void DefaultOptions_ShouldHaveRepetitionPenalty1()
    {
        var options = new TranslatorOptions();

        options.RepetitionPenalty.Should().Be(1.0f);
    }

    [Fact]
    public void Clone_ShouldCopyBeamSearchSettings()
    {
        var original = new TranslatorOptions
        {
            BeamWidth = 8,
            UseGreedyDecoding = true,
            LengthPenalty = 1.5f,
            RepetitionPenalty = 1.2f
        };

        var clone = original.Clone();

        clone.BeamWidth.Should().Be(8);
        clone.UseGreedyDecoding.Should().BeTrue();
        clone.LengthPenalty.Should().Be(1.5f);
        clone.RepetitionPenalty.Should().Be(1.2f);
    }

    [Fact]
    public void Clone_ShouldNotAffectOriginal()
    {
        var original = new TranslatorOptions
        {
            BeamWidth = 4,
            LengthPenalty = 1.0f
        };

        var clone = original.Clone();
        clone.BeamWidth = 8;
        clone.LengthPenalty = 1.5f;

        original.BeamWidth.Should().Be(4);
        original.LengthPenalty.Should().Be(1.0f);
    }
}
