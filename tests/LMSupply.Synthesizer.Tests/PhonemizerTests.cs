using AwesomeAssertions;
using LMSupply.Synthesizer.Core;
using LMSupply.Synthesizer.Core.English;
using LMSupply.Synthesizer.Models;

namespace LMSupply.Synthesizer.Tests;

/// <summary>
/// Text reaches a voice through a phonemizer: the caller's, or a built-in one — the text itself for a voice whose phonemes
/// are its text, the English phonemizer for an English voice. A voice with none of them is refused, by an exception that
/// says what the voice takes and how to supply it.
/// </summary>
public class PhonemizerTests
{
    private static readonly VoiceConfig Korean =
        VoiceFixture.Minimal("\"espeak\": {\"voice\": \"ko\"}, \"language\": {\"code\": \"ko_KR\"}", catalog: SynthesizerModels.KoKr);
    private static readonly VoiceConfig English = VoiceFixture.LjSpeech();
    private static readonly VoiceConfig Text = VoiceFixture.Minimal("\"phoneme_type\": \"text\"");

    [Fact]
    public void TextForAVoiceThatNeedsAPhonemizer_IsRefused_NamingWhatTheVoiceTakesAndTheTwoWaysOut()
    {
        var act = () => OnnxSynthesizerModel.ResolvePhonemizer(callers: null, Korean, "korean");

        var message = act.Should().Throw<NotSupportedException>().Which.Message;
        message.Should().Contain("'korean'").And.Contain("'espeak'").And.Contain("'ko'").And.Contain("'ko-KR'");
        message.Should().Contain("English voices only");
        message.Should().Contain("SynthesizerOptions.Phonemizer").And.Contain("SynthesizePhonemesAsync");
        message.Should().MatchRegex(@"^[\x20-\x7E]+$", "operational text is ASCII");
    }

    [Fact]
    public void AVoiceThatNamesNoPhonemizerVoice_IsStillRefusedWithAReadableMessage()
    {
        var voice = VoiceFixture.Minimal(catalog: SynthesizerModels.KoKr);
        var act = () => OnnxSynthesizerModel.ResolvePhonemizer(callers: null, voice, "korean");

        act.Should().Throw<NotSupportedException>().WithMessage("*'espeak'*'unspecified'*'ko-KR'*");
    }

    [Fact]
    public void TheCallersPhonemizer_IsUsedForAnyVoice()
    {
        var callers = new FixedPhonemizer();

        OnnxSynthesizerModel.ResolvePhonemizer(callers, Korean, "korean").Should().BeSameAs(callers);
        OnnxSynthesizerModel.ResolvePhonemizer(callers, English, "default").Should().BeSameAs(callers);
        OnnxSynthesizerModel.ResolvePhonemizer(callers, Text, "default").Should().BeSameAs(callers);
    }

    [Fact]
    public void AVoiceWhosePhonemesAreItsText_NeedsNoPhonemizer()
    {
        OnnxSynthesizerModel.ResolvePhonemizer(callers: null, Text, "default").Should().BeOfType<CodepointPhonemizer>();
    }

    [Fact]
    public async Task TextPhonemes_AreTheLowerCasedDecomposedCharacters_AsOneSentence()
    {
        // U+00C7 (capital c-cedilla) lower-cases to U+00E7 and decomposes to 'c' + U+0327.
        var text = (char)0x00C7 + "a Va. Bien!";

        var sentences = await CodepointPhonemizer.Instance.PhonemizeAsync(text, Text.Scheme, TestContext.Current.CancellationToken);

        sentences.Should().Equal("c" + (char)0x0327 + "a va. bien!");
    }

    [Fact]
    public async Task TextPhonemes_EncodeThroughTheVoicesOwnMap()
    {
        var voice = VoiceFixture.Minimal("\"phoneme_type\": \"text\"");

        var sentences = await OnnxSynthesizerModel.ResolvePhonemizer(callers: null, voice, "default")
            .PhonemizeAsync("AB", voice.Scheme, TestContext.Current.CancellationToken);

        voice.Scheme.Type.Should().Be("text");
        voice.Encoder.Encode(sentences.Single()).Ids.Should().Equal(1, 0, 3, 0, 4, 0, 2);
    }

    [Fact]
    public void AnEnglishVoice_GetsTheBuiltInEnglishPhonemizer()
    {
        OnnxSynthesizerModel.ResolvePhonemizer(callers: null, English, "default").Should().BeOfType<EnglishPhonemizer>();
    }

    private sealed class FixedPhonemizer : IPhonemizer
    {
        public ValueTask<IReadOnlyList<string>> PhonemizeAsync(string text, PhonemeScheme scheme, CancellationToken cancellationToken = default)
            => new(["ab"]);
    }
}
