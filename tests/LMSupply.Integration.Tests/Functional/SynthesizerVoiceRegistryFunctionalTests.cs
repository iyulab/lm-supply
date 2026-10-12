using AwesomeAssertions;
using LMSupply.Synthesizer;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// The voices the registry changed in 0.88.0 load and produce audio: the new default (LJ Speech) and the Korean voice, which
/// used to name an x_low build that was never published. Loading only — whether the audio is speech is
/// <see cref="SynthesizerIntelligibilityFacts"/>'s question. The phonemes are rough, hand-written ones for each voice.
/// </summary>
[Trait("Category", "Integration")]
public sealed class SynthesizerVoiceRegistryFunctionalTests
{
    [Theory]
    [InlineData("default", "həlˈəʊ, ðɪs ɪz ɐ tˈɛst.", "espeak", "en")]
    [InlineData("korean", "ˈɐnnjʌŋhˌɐsejˌo.", "espeak", "ko")]
    public async Task Voice_LoadsAndProducesAudio(string alias, string phonemes, string phonemeType, string phonemizerVoice)
    {
        await using var model = await LocalSynthesizer.LoadAsync(alias, cancellationToken: TestContext.Current.CancellationToken);

        var result = await model.SynthesizePhonemesAsync([phonemes], cancellationToken: TestContext.Current.CancellationToken);

        model.PhonemeScheme.Type.Should().Be(phonemeType);
        model.PhonemeScheme.Voice.Should().Be(phonemizerVoice);
        result.AudioSamples.Should().NotBeEmpty();
        result.ToWavBytes().Length.Should().BeGreaterThan(44, "a WAV longer than its header");
    }

    /// <summary>
    /// The catalog's voices share a repository id, so a voice is loaded by its catalog key. The repository id on its own
    /// names no voice and loads the first of the catalog, the default one.
    /// </summary>
    [Theory]
    [InlineData("rhasspy/piper-voices:ko/ko_KR/kss/medium", "ko/ko_KR/kss/medium", "ko")]
    [InlineData("rhasspy/piper-voices", "en/en_US/ljspeech/medium", "en")]
    public async Task Voice_LoadsByCatalogKey(string modelId, string voiceFolder, string phonemizerVoice)
    {
        await using var model = await LocalSynthesizer.LoadAsync(modelId, cancellationToken: TestContext.Current.CancellationToken);

        model.Voice.Should().Be(voiceFolder);
        model.PhonemeScheme.Voice.Should().Be(phonemizerVoice);
    }

    /// <summary>
    /// A voice with several speakers is run with the speaker input its graph requires, by the count its config declares.
    /// </summary>
    [Fact]
    public async Task AMultiSpeakerVoice_SpeaksWithEachOfItsSpeakers()
    {
        await using var model = await LocalSynthesizer.LoadAsync("british", cancellationToken: TestContext.Current.CancellationToken);
        string[] phonemes = ["həlˈəʊ, ðɪs ɪz ɐ tˈɛst."];

        var first = await model.SynthesizePhonemesAsync(phonemes, cancellationToken: TestContext.Current.CancellationToken);
        var last = await model.SynthesizePhonemesAsync(phonemes, new SynthesizeOptions { SpeakerId = 3 }, TestContext.Current.CancellationToken);
        var beyond = () => model.SynthesizePhonemesAsync(phonemes, new SynthesizeOptions { SpeakerId = 4 }, TestContext.Current.CancellationToken);

        model.GetModelInfo()!.NumSpeakers.Should().Be(4);
        first.AudioSamples.Should().NotBeEmpty();
        last.AudioSamples.Should().NotBeEmpty();
        await beyond.Should().ThrowAsync<ArgumentOutOfRangeException>();
    }
}
