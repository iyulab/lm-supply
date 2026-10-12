using AwesomeAssertions;
using LMSupply.Exceptions;
using LMSupply.Synthesizer.Core;
using LMSupply.Synthesizer.Models;

namespace LMSupply.Synthesizer.Tests;

/// <summary>
/// A voice is run by what its config declares: phoneme type, phonemizer voice and language, inference defaults, sample
/// rate and speaker count. The catalog entry fills in only what a config leaves out.
/// </summary>
public class VoiceConfigTests
{
    [Fact]
    public void TheDefaultVoicesConfig_IsReadAsDeclared()
    {
        var voice = VoiceFixture.LjSpeech();

        voice.Scheme.Type.Should().Be("espeak");
        voice.Scheme.Voice.Should().Be("en");
        voice.Scheme.Language.Should().Be("en-US");
        voice.SampleRate.Should().Be(22050);
        voice.NumSpeakers.Should().Be(1);
        voice.NoiseScale.Should().Be(0.667f);
        voice.LengthScale.Should().Be(1.0f);
        voice.NoiseWidth.Should().Be(0.333f, "this voice declares its own noise_w, not Piper's general 0.8");
    }

    [Fact]
    public void AConfigThatNamesNoPhonemeType_IsEspeak()
    {
        VoiceFixture.Minimal().Scheme.Type.Should().Be("espeak");
    }

    [Fact]
    public void TheConfigsSampleRateAndSpeakerCount_WinOverTheCatalogEntrys()
    {
        var catalog = DefaultModels.EnUsLjSpeech with { SampleRate = 22050, NumSpeakers = 1 };

        var voice = VoiceFixture.Minimal("\"audio\": {\"sample_rate\": 16000}, \"num_speakers\": 4", catalog: catalog);
        var reported = voice.Reconcile(catalog);

        voice.SampleRate.Should().Be(16000);
        voice.NumSpeakers.Should().Be(4);
        reported.SampleRate.Should().Be(16000);
        reported.NumSpeakers.Should().Be(4);
        reported.AliasName.Should().Be(catalog.AliasName);
    }

    [Fact]
    public void WhatAConfigLeavesOut_ComesFromTheCatalogEntryOrPipersDefaults()
    {
        var catalog = DefaultModels.EnUsLjSpeech with { SampleRate = 16000, Language = "en-GB" };

        var voice = VoiceFixture.Minimal(catalog: catalog);

        voice.SampleRate.Should().Be(16000);
        voice.Scheme.Language.Should().Be("en-GB");
        voice.Scheme.Voice.Should().BeNull();
        voice.NumSpeakers.Should().Be(1);
        voice.Scales(null).Should().Equal(0.667f, 1.0f, 0.8f);
        voice.Reconcile(catalog).Should().BeSameAs(catalog);
    }

    [Fact]
    public void Scales_AreTheVoicesOwnDefaults_WhenTheCallerSetsNone()
    {
        var voice = VoiceFixture.LjSpeech();

        voice.Scales(null).Should().Equal(0.667f, 1.0f, 0.333f);
        voice.Scales(new SynthesizeOptions()).Should().Equal(0.667f, 1.0f, 0.333f);
    }

    [Fact]
    public void Scales_TakeTheCallersNoiseValues_WhereSet()
    {
        var scales = VoiceFixture.LjSpeech().Scales(new SynthesizeOptions { NoiseScale = 0.5f, NoiseWidth = 0.9f });

        scales.Should().Equal(0.5f, 1.0f, 0.9f);
    }

    [Theory]
    [InlineData(2.0f, 0.5f)]   // twice as fast: half the length
    [InlineData(0.5f, 2.0f)]   // half as fast: twice the length
    [InlineData(1.0f, 1.0f)]
    public void Speed_DividesTheVoicesLengthScale(float speed, float expectedLengthScale)
    {
        var scales = VoiceFixture.LjSpeech().Scales(new SynthesizeOptions { Speed = speed });

        scales[1].Should().Be(expectedLengthScale);
    }

    [Fact]
    public void Speed_IsRelativeToTheLengthScaleTheVoiceDeclares()
    {
        var voice = VoiceFixture.Minimal("\"inference\": {\"length_scale\": 1.5}");

        voice.Scales(null)[1].Should().Be(1.5f);
        voice.Scales(new SynthesizeOptions { Speed = 3.0f })[1].Should().Be(0.5f);
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(-1f)]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    public void ASpeedThatIsNotPositiveAndFinite_IsRefused(float speed)
    {
        var act = () => VoiceFixture.LjSpeech().Scales(new SynthesizeOptions { Speed = speed });

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void ASingleSpeakerVoice_TakesNoSpeakerInput()
    {
        var voice = VoiceFixture.LjSpeech();

        voice.SpeakerId(null).Should().BeNull();
        voice.SpeakerId(new SynthesizeOptions { SpeakerId = 3 }).Should().BeNull("Piper ignores the speaker for a single-speaker voice");
    }

    [Fact]
    public void AMultiSpeakerVoice_TakesTheCallersSpeaker_AndTheFirstByDefault()
    {
        var voice = VoiceFixture.Minimal("\"num_speakers\": 4");

        voice.SpeakerId(null).Should().Be(0);
        voice.SpeakerId(new SynthesizeOptions { SpeakerId = 3 }).Should().Be(3);
    }

    [Theory]
    [InlineData(4)]
    [InlineData(-1)]
    public void ASpeakerTheVoiceDoesNotHave_IsRefused(int speakerId)
    {
        var voice = VoiceFixture.Minimal("\"num_speakers\": 4");

        var act = () => voice.SpeakerId(new SynthesizeOptions { SpeakerId = speakerId });

        act.Should().Throw<ArgumentOutOfRangeException>().WithMessage("*between 0 and 3*");
    }

    [Theory]
    [InlineData("{}")]
    [InlineData("{\"phoneme_id_map\": {}}")]
    public void AConfigWithoutPhonemeIds_FailsTheLoad(string json)
    {
        var act = () => VoiceConfig.Parse(json, DefaultModels.EnUsLjSpeech);

        act.Should().Throw<ModelLoadException>().WithMessage("*no phoneme_id_map*");
    }

    [Theory]
    [InlineData("\"^\": [1], \"$\": [2]", "'_'")]
    [InlineData("\"_\": [0], \"$\": [2]", "'^'")]
    [InlineData("\"_\": [0], \"^\": [1]", "'$'")]
    public void AConfigWithoutThePadStartOrEndSymbol_FailsTheLoad(string idMap, string missing)
    {
        var act = () => VoiceConfig.Parse("{\"phoneme_id_map\": {" + idMap + ", \"a\": [3]}}", DefaultModels.EnUsLjSpeech);

        act.Should().Throw<ModelLoadException>().WithMessage($"*no id for {missing}*");
    }

    [Fact]
    public void AConfigThatIsNotJson_FailsTheLoad()
    {
        var act = () => VoiceConfig.Parse("not json", DefaultModels.EnUsLjSpeech);

        act.Should().Throw<ModelLoadException>().WithMessage("*not valid JSON*");
    }
}
