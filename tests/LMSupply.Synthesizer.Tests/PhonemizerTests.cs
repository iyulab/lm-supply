using AwesomeAssertions;
using LMSupply.Synthesizer.Core;

namespace LMSupply.Synthesizer.Tests;

/// <summary>
/// Text reaches a voice through a phonemizer: the caller's, or the built-in one for a voice whose phonemes are its text.
/// A voice with neither is refused, by an exception that says what the voice takes and how to supply it.
/// </summary>
public class PhonemizerTests
{
    private static readonly PhonemeScheme Espeak = new() { Type = "espeak", Voice = "en-us", Language = "en-US" };
    private static readonly PhonemeScheme Text = new() { Type = "text", Language = "en-US" };

    [Fact]
    public void TextForAVoiceThatNeedsAPhonemizer_IsRefused_NamingWhatTheVoiceTakesAndTheTwoWaysOut()
    {
        var act = () => OnnxSynthesizerModel.ResolvePhonemizer(callers: null, Espeak, "default");

        var message = act.Should().Throw<NotSupportedException>().Which.Message;
        message.Should().Contain("'default'").And.Contain("'espeak'").And.Contain("'en-us'").And.Contain("'en-US'");
        message.Should().Contain("SynthesizerOptions.Phonemizer").And.Contain("SynthesizePhonemesAsync");
        message.Should().MatchRegex(@"^[\x20-\x7E]+$", "operational text is ASCII");
    }

    [Fact]
    public void ASchemeThatNamesNoVoiceOrLanguage_IsStillRefusedWithAReadableMessage()
    {
        var act = () => OnnxSynthesizerModel.ResolvePhonemizer(callers: null, new PhonemeScheme { Type = "espeak" }, "default");

        act.Should().Throw<NotSupportedException>().WithMessage("*'espeak'*'unspecified'*");
    }

    [Fact]
    public void TheCallersPhonemizer_IsUsedForAnyVoice()
    {
        var callers = new FixedPhonemizer();

        OnnxSynthesizerModel.ResolvePhonemizer(callers, Espeak, "default").Should().BeSameAs(callers);
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

        var sentences = await CodepointPhonemizer.Instance.PhonemizeAsync(text, Text, TestContext.Current.CancellationToken);

        sentences.Should().Equal("c" + (char)0x0327 + "a va. bien!");
    }

    [Fact]
    public async Task TextPhonemes_EncodeThroughTheVoicesOwnMap()
    {
        var voice = VoiceFixture.Minimal("\"phoneme_type\": \"text\"");

        var sentences = await OnnxSynthesizerModel.ResolvePhonemizer(callers: null, voice.Scheme, "default")
            .PhonemizeAsync("AB", voice.Scheme, TestContext.Current.CancellationToken);

        voice.Scheme.Type.Should().Be("text");
        voice.Encoder.Encode(sentences.Single()).Ids.Should().Equal(1, 0, 3, 0, 4, 0, 2);
    }

    private sealed class FixedPhonemizer : IPhonemizer
    {
        public ValueTask<IReadOnlyList<string>> PhonemizeAsync(string text, PhonemeScheme scheme, CancellationToken cancellationToken = default)
            => new(["ab"]);
    }
}
