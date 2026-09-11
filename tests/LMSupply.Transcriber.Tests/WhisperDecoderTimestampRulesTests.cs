using AwesomeAssertions;
using LMSupply.Transcriber.Decoding;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// Tests for the timestamp rules <see cref="WhisperDecoder.SelectNextToken"/> applies when segment
/// timestamps are requested (<see cref="TranscribeOptions.WordTimestamps"/>). They mirror the
/// reference decoder's <c>ApplyTimestampRules</c>: without them a greedy decoder almost never picks a
/// timestamp token after the first, so "timestamp mode" returned one segment spanning the whole window
/// — the segment boundaries the option documents did not exist, and a segment's end could not tell a
/// caller where the speech it holds actually stopped.
/// </summary>
public class WhisperDecoderTimestampRulesTests
{
    private const int EndOfText = 50257;
    private const int StartOfTranscript = 50258;
    private const int Transcribe = 50359;
    private const int NoTimestamps = 50363;
    private const int TimestampBegin = 50364;
    private const int VocabSize = 51865;
    private const int TextA = 440;
    private const int TextB = 264;

    private static readonly int[] Prompt = [StartOfTranscript, Transcribe];
    private static readonly TranscribeOptions WithTimestamps = new() { WordTimestamps = true };

    private static int Ts(double seconds) => TimestampBegin + (int)Math.Round(seconds / 0.02);

    private static float[] Logits()
    {
        var logits = new float[VocabSize];
        Array.Fill(logits, -1000f);
        return logits;
    }

    private static WhisperDecoder Decoder() => WhisperDecoder.CreateForTesting(WhisperTokenizer.CreateDefault());

    private static List<int> Generated(params int[] tokens) => [.. Prompt, .. tokens];

    [Fact]
    public void FirstStep_IsATimestampWithinTheFirstSecond()
    {
        var logits = Logits();
        logits[TextA] = 10f;       // text would win an unconstrained argmax
        logits[Ts(5.0)] = 9f;      // a timestamp past the initial window
        logits[Ts(0.0)] = 1f;

        var selected = Decoder().SelectNextToken(logits, Generated(), Prompt, WithTimestamps);

        selected.Should().Be(Ts(0.0));
    }

    [Fact]
    public void TimestampProbabilityMassAboveTheBestTextToken_ForcesATimestamp()
    {
        // No single timestamp beats the text token, but together they carry more probability — the
        // reference rule that produces segment boundaries at all.
        var logits = Logits();
        logits[TextB] = 2.0f;
        for (var t = 1.0; t < 1.4; t += 0.02)
        {
            logits[Ts(t)] = 0.5f;
        }

        var selected = Decoder().SelectNextToken(logits, Generated(Ts(0.0), TextA), Prompt, WithTimestamps);

        selected.Should().BeGreaterThanOrEqualTo(TimestampBegin);
    }

    [Fact]
    public void AfterATimestampPair_TheNextTokenIsText()
    {
        var logits = Logits();
        logits[Ts(3.0)] = 10f;
        logits[TextB] = 1f;

        var selected = Decoder().SelectNextToken(logits, Generated(Ts(0.0), TextA, Ts(2.0), Ts(2.0)), Prompt, WithTimestamps);

        selected.Should().Be(TextB);
    }

    [Fact]
    public void AfterAClosingTimestamp_TheNextTokenIsNotText()
    {
        var logits = Logits();
        logits[TextB] = 10f;
        logits[Ts(2.0)] = 1f;

        var selected = Decoder().SelectNextToken(logits, Generated(Ts(0.0), TextA, Ts(2.0)), Prompt, WithTimestamps);

        selected.Should().Be(Ts(2.0), "a closing timestamp is followed by the next segment's opening one (or end of text)");
    }

    [Fact]
    public void Timestamps_NeverGoBackwards()
    {
        var logits = Logits();
        logits[Ts(1.0)] = 10f;     // earlier than a timestamp already emitted
        logits[Ts(5.0)] = 5f;

        var selected = Decoder().SelectNextToken(logits, Generated(Ts(0.0), TextA, Ts(4.0), Ts(4.0), TextB), Prompt, WithTimestamps);

        selected.Should().Be(Ts(5.0));
    }

    [Fact]
    public void NoTimestampsToken_IsNeverSelectedInTimestampMode()
    {
        var logits = Logits();
        logits[NoTimestamps] = 10f;
        logits[TextB] = 1f;

        var selected = Decoder().SelectNextToken(logits, Generated(Ts(0.0), TextA), Prompt, WithTimestamps);

        selected.Should().NotBe(NoTimestamps);
    }

    [Fact]
    public void WithoutTimestamps_TheRulesDoNotApply()
    {
        var logits = Logits();
        logits[TextA] = 10f;
        logits[Ts(0.0)] = 9f;

        var selected = Decoder().SelectNextToken(logits, Generated(NoTimestamps), Prompt, options: null);

        selected.Should().Be(TextA);
    }
}
