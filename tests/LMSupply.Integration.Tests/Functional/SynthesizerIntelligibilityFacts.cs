using LMSupply.Integration.Tests.Helpers;
using LMSupply.Synthesizer;
using LMSupply.Transcriber;

namespace LMSupply.Integration.Tests.Functional;

/// <summary>
/// The synthesizer speaks: what it says, Whisper hears. This is the positive control the other synthesizer facts lack — «audio
/// came out» holds for noise too.
/// </summary>
/// <remarks>
/// A voice is run by the contract its config declares, so phonemes of the voice's scheme come out as speech, whether the
/// caller passes them or a phonemizer set in the options produces them. The package has no phonemizer of its own for the
/// catalog's voices; text without one is refused, never spoken as noise. When a built-in phonemizer lands, the refusal
/// facts for its languages become intelligibility facts for plain text.
/// </remarks>
[Trait("Category", "Integration")]
[Trait("Category", "Intelligibility")]
public sealed class SynthesizerIntelligibilityFacts(ITestOutputHelper output)
{
    // Measured with the default voice and the default transcriber: both facts' transcripts matched word for word (word
    // error rate 0; the facts write what they measure to the test output). The bound leaves room for two of the fifteen
    // words. For comparison: with ids that are not the voice's own the rate measured was 1.0, and without the pad after
    // each phoneme 0.93. Swapping the start and end symbols is not audible; the encoder's unit tests pin those.
    private const double MaxWordErrorRate = 0.15;

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    [Fact]
    public async Task PhonemesTheCallerPasses_WhisperHears()
    {
        await using var synthesizer = await LocalSynthesizer.LoadAsync("default", cancellationToken: Ct);

        var audio = await synthesizer.SynthesizePhonemesAsync([SpokenPhonemes.Weather, SpokenPhonemes.Fox], cancellationToken: Ct);

        audio.UnknownPhonemes.Should().BeEmpty("the voice has an id for every phoneme of these sentences");
        await AssertHeardAsync($"{SpokenPhonemes.WeatherText} {SpokenPhonemes.FoxText}", audio);
    }

    [Fact]
    public async Task TextThroughThePhonemizerInTheOptions_WhisperHears()
    {
        var phonemizer = new SpokenPhonemes.Phonemizer();
        var options = new SynthesizerOptions { Phonemizer = phonemizer };
        await using var synthesizer = await LocalSynthesizer.LoadAsync("default", options, cancellationToken: Ct);
        var text = $"{SpokenPhonemes.WeatherText} {SpokenPhonemes.FoxText}";

        var audio = await synthesizer.SynthesizeAsync(text, cancellationToken: Ct);

        // The phonemizer is asked for what the voice's own config declares.
        phonemizer.Schemes.Should().ContainSingle().Which.Should().Be(synthesizer.PhonemeScheme);
        synthesizer.PhonemeScheme.Should().Be(new PhonemeScheme { Type = "espeak", Voice = "en", Language = "en-US" });
        audio.Text.Should().Be(text);
        audio.Phonemes.Should().Equal(SpokenPhonemes.Weather, SpokenPhonemes.Fox);
        await AssertHeardAsync(text, audio);
    }

    [Theory]
    [InlineData("default", "The weather is beautiful today.", "'espeak'", "'en-US'")]
    [InlineData("korean", "오늘 날씨가 정말 좋습니다.", "'espeak'", "'ko-KR'")]
    public async Task TextWithoutAPhonemizer_IsRefused_NotSpokenAsNoise(string alias, string text, string phonemeType, string language)
    {
        await using var synthesizer = await LocalSynthesizer.LoadAsync(alias, cancellationToken: Ct);
        using var stream = new MemoryStream();
        var file = Path.Combine(Path.GetTempPath(), $"lmsupply-refused-{Guid.NewGuid():N}.wav");

        var whole = () => synthesizer.SynthesizeAsync(text, cancellationToken: Ct);
        var toStream = () => synthesizer.SynthesizeToStreamAsync(text, stream, cancellationToken: Ct);
        var toFile = () => synthesizer.SynthesizeToFileAsync(text, file, cancellationToken: Ct);
        var streaming = async () =>
        {
            await foreach (var _ in synthesizer.SynthesizeStreamingAsync(text, cancellationToken: Ct)) { }
        };

        (await whole.Should().ThrowAsync<NotSupportedException>())
            .WithMessage($"*{phonemeType}*").WithMessage($"*{language}*")
            .WithMessage("*SynthesizerOptions.Phonemizer*").WithMessage("*SynthesizePhonemesAsync*");
        await toStream.Should().ThrowAsync<NotSupportedException>();
        await toFile.Should().ThrowAsync<NotSupportedException>();
        await streaming.Should().ThrowAsync<NotSupportedException>();
        stream.Length.Should().Be(0, "a refusal writes no audio");
        File.Exists(file).Should().BeFalse("a refusal leaves no file behind");
    }

    private async Task AssertHeardAsync(string said, SynthesisResult audio)
    {
        await using var transcriber = await LocalTranscriber.LoadAsync("default", cancellationToken: Ct);
        var heard = await transcriber.TranscribeAsync(audio.ToWavBytes(), new TranscribeOptions { Language = "en" }, Ct);

        var wer = WordErrorRate(Words(said), Words(heard.Text));
        output.WriteLine($"said «{said}», Whisper heard «{heard.Text.Trim()}» ({audio.DurationSeconds:F1} s) — word error rate {wer:F3}");

        Assert.True(wer <= MaxWordErrorRate,
            $"said «{said}», Whisper heard «{heard.Text}» ({audio.DurationSeconds:F1} s) — word error rate {wer:F3}, bound {MaxWordErrorRate}");
    }

    private static string[] Words(string text) =>
        [.. new string([.. text.ToLowerInvariant().Select(c => char.IsLetterOrDigit(c) || c == '\'' ? c : ' ')])
            .Split(' ', StringSplitOptions.RemoveEmptyEntries)];

    // Word-level edit distance (substitutions, insertions, deletions) over the number of words said.
    private static double WordErrorRate(string[] said, string[] heard)
    {
        var distance = new int[said.Length + 1, heard.Length + 1];
        for (var i = 0; i <= said.Length; i++) distance[i, 0] = i;
        for (var j = 0; j <= heard.Length; j++) distance[0, j] = j;

        for (var i = 1; i <= said.Length; i++)
        {
            for (var j = 1; j <= heard.Length; j++)
            {
                var substitution = distance[i - 1, j - 1] + (said[i - 1] == heard[j - 1] ? 0 : 1);
                distance[i, j] = Math.Min(substitution, Math.Min(distance[i - 1, j] + 1, distance[i, j - 1] + 1));
            }
        }

        return (double)distance[said.Length, heard.Length] / said.Length;
    }
}
