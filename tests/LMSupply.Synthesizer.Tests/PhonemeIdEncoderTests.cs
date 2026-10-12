using AwesomeAssertions;

namespace LMSupply.Synthesizer.Tests;

/// <summary>
/// Phonemes become the ids the voice was trained on: start, pad, then every phoneme followed by a pad, then end. The
/// expected ids are written out from the default voice's own <c>phoneme_id_map</c> (<c>_</c> 0, <c>^</c> 1, <c>$</c> 2),
/// so an encoder that drops the pads, swaps the ends or uses ids of its own fails on the exact array.
/// </summary>
public class PhonemeIdEncoderTests
{
    // "the" as eSpeak writes it: U+00F0 (id 41), U+0259 (id 59).
    private const string The = "ðə";

    [Fact]
    public void Phonemes_AreEncodedAsStartPadThenEachPhonemeAndAPadThenEnd()
    {
        var encoded = VoiceFixture.LjSpeech().Encoder.Encode(The);

        encoded.Ids.Should().Equal(1, 0, 41, 0, 59, 0, 2);
        encoded.PhonemeCount.Should().Be(2);
        encoded.Unknown.Should().BeEmpty();
    }

    [Fact]
    public void ASpaceAndClausePunctuation_ArePhonemesToo()
    {
        // ' ' is id 3, ',' 8 and '.' 10 in the voice's map.
        var encoded = VoiceFixture.LjSpeech().Encoder.Encode(The + ", " + The + ".");

        encoded.Ids.Should().Equal(1, 0, 41, 0, 59, 0, 8, 0, 3, 0, 41, 0, 59, 0, 10, 0, 2);
        encoded.PhonemeCount.Should().Be(7);
    }

    [Fact]
    public void NoPhonemes_EncodeAsStartPadEnd_AndCountAsNone()
    {
        var encoded = VoiceFixture.LjSpeech().Encoder.Encode("");

        encoded.Ids.Should().Equal(1, 0, 2);
        encoded.PhonemeCount.Should().Be(0);
    }

    [Fact]
    public void APhonemeTheVoiceHasNoIdFor_IsSkippedAndReported()
    {
        // The euro sign is not a phoneme of this voice; the two around it are.
        var encoded = VoiceFixture.LjSpeech().Encoder.Encode("ð" + (char)0x20AC + "ə" + (char)0x20AC);

        encoded.Ids.Should().Equal(1, 0, 41, 0, 59, 0, 2);
        encoded.PhonemeCount.Should().Be(2);
        encoded.Unknown.Should().Be(new string((char)0x20AC, 2));
    }

    [Fact]
    public void Phonemes_AreDecomposedBeforeTheyAreLookedUp()
    {
        // The voice has an id for the precomposed c-cedilla (U+00E7: 40) and for the combining cedilla (U+0327: 140).
        // Piper decomposes first, so the precomposed letter is 'c' (16) and the mark (140), never 40.
        var precomposed = ((char)0x00E7).ToString();
        var decomposed = "c" + (char)0x0327;

        var encoder = VoiceFixture.LjSpeech().Encoder;

        encoder.Encode(precomposed).Ids.Should().Equal(1, 0, 16, 0, 140, 0, 2);
        encoder.Encode(decomposed).Ids.Should().Equal(1, 0, 16, 0, 140, 0, 2);
    }

    [Fact]
    public void APhonemeWithSeveralIds_ContributesAllOfThemBeforeItsPad()
    {
        var encoded = VoiceFixture.Minimal(extraPhonemes: "\"x\": [7, 8]").Encoder.Encode("axb");

        encoded.Ids.Should().Equal(1, 0, 3, 0, 7, 8, 0, 4, 0, 2);
        encoded.PhonemeCount.Should().Be(3);
    }

    [Fact]
    public void APhoneme_IsOneCodePoint_WhateverItsUtf16Length()
    {
        // U+1D11E takes two UTF-16 code units; it is one phoneme with one id, not two unknown halves.
        var astral = char.ConvertFromUtf32(0x1D11E);

        var encoded = VoiceFixture.Minimal(extraPhonemes: "\"" + astral + "\": [9]").Encoder.Encode("a" + astral);

        encoded.Ids.Should().Equal(1, 0, 3, 0, 9, 0, 2);
        encoded.Unknown.Should().BeEmpty();
    }

    [Fact]
    public void AMapKeyOfSeveralCodePoints_IsNeverMatched_AndDoesNotStopTheVoiceLoading()
    {
        // Published configs carry such keys (diphthongs). Phonemes are looked up one code point at a time.
        var encoded = VoiceFixture.Minimal(extraPhonemes: "\"ab\": [9]").Encoder.Encode("ab");

        encoded.Ids.Should().Equal(1, 0, 3, 0, 4, 0, 2);
    }

    [Fact]
    public void EveryPhonemeOfAnEspeakSentence_HasAnIdInTheDefaultVoice()
    {
        // "The weather is beautiful today." with stress and length marks, a linking r and the final full stop.
        var encoded = VoiceFixture.LjSpeech().Encoder.Encode("ðə wˈɛðəɹ ɪz bjˈuːtifəl tədˈeɪ.");

        encoded.Unknown.Should().BeEmpty();
        encoded.PhonemeCount.Should().Be(31);
        encoded.Ids.Should().HaveCount(2 + 31 * 2 + 1);
    }
}
