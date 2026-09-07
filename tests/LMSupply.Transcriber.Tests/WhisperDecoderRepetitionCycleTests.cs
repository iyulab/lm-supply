using AwesomeAssertions;
using LMSupply.Transcriber.Decoding;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// Tests for <see cref="WhisperDecoder.TryDetectRepetitionCycle"/> — the tail check that ends a
/// chunk once greedy decoding has collapsed into a periodic token cycle. Pure list logic, no ONNX
/// session involved.
/// </summary>
public class WhisperDecoderRepetitionCycleTests
{
    private const int StartOfTranscriptToken = 50258;
    private const int TranscribeToken = 50359;
    private const int NoTimestampsToken = 50363;
    private const int TheUpper = 440;   // " The"
    private const int TheLower = 264;   // " the"

    private const int GeneratedStart = 2; // [SOT, Transcribe]

    /// <summary>
    /// Rebuilds the generated tokens captured in docket iyulab/lm-supply#59's 2026-09-07 fp32
    /// trace (LMSupply.Transcriber 0.55.0, whisper-base, CPU EP, real fixture): NoTimestamps, a
    /// three-token stutter, then a period-4 cycle from step 4 onward.
    /// </summary>
    private static List<int> CapturedTrace(int steps)
    {
        var generated = new List<int> { NoTimestampsToken, TheUpper, TheUpper, TheUpper };
        while (generated.Count < steps)
        {
            generated.Add(generated.Count % 4 == 3 ? TheUpper : TheLower);
        }

        return [StartOfTranscriptToken, TranscribeToken, .. generated.Take(steps)];
    }

    [Fact]
    public void TryDetectRepetitionCycle_CapturedPeriod4Trace_EndsTheChunk()
    {
        // The captured failure: from step 4 the decoder cycles with period 4 (three " the", one
        // " The"). SelectNextToken's three-identical-token guard cannot see that — and in fact
        // sustains it, by suppressing " the" at exactly every third occurrence. Left alone the
        // decoder ran to _maxLength (448) emitting no timestamp token at all, which is why the
        // reported chunk yielded zero usable segments.
        var tokens = CapturedTrace(15);

        var detected = WhisperDecoder.TryDetectRepetitionCycle(tokens, GeneratedStart, out var cycleStart);

        detected.Should().BeTrue("a period-4 unit repeated three times over twelve tokens is a " +
            "decoder collapse, not speech");
        cycleStart.Should().BeLessThan(tokens.Count, "a cycle must start before the current tail");
        tokens.Skip(cycleStart).Should().OnlyContain(t => t == TheLower || t == TheUpper,
            "the discarded span must be exactly the degenerate cycle, not real text preceding it");
    }

    [Fact]
    public void TryDetectRepetitionCycle_ShorterPrefixOfSameTrace_DoesNotFireYet()
    {
        // Two turns of that same cycle is not yet enough evidence — the check waits for a third.
        var tokens = CapturedTrace(12);

        WhisperDecoder.TryDetectRepetitionCycle(tokens, GeneratedStart, out _)
            .Should().BeFalse();
    }

    [Fact]
    public void TryDetectRepetitionCycle_SingleStutter_DoesNotFire()
    {
        // The exact state the 0.42.6 EOT penalty was built to decode *past*: three identical
        // tokens in a row. Ending a chunk here would undo that fix, so the thresholds must not
        // treat a lone stutter as a cycle.
        var tokens = new List<int>
        {
            StartOfTranscriptToken, TranscribeToken, NoTimestampsToken, TheUpper, TheUpper, TheUpper
        };

        WhisperDecoder.TryDetectRepetitionCycle(tokens, GeneratedStart, out _)
            .Should().BeFalse();
    }

    [Fact]
    public void TryDetectRepetitionCycle_TwelveIdenticalTokens_FiresOnPeriodOne()
    {
        // A period-1 tail still qualifies, but only at the same twelve-token weight of evidence.
        var tokens = new List<int> { StartOfTranscriptToken, TranscribeToken, NoTimestampsToken };
        tokens.AddRange(Enumerable.Repeat(TheUpper, 12));

        var detected = WhisperDecoder.TryDetectRepetitionCycle(tokens, GeneratedStart, out var cycleStart);

        detected.Should().BeTrue();
        tokens.Skip(cycleStart).Should().OnlyContain(t => t == TheUpper);
    }

    [Fact]
    public void TryDetectRepetitionCycle_OrdinaryVariedSpeech_DoesNotFire()
    {
        // A long tail of distinct tokens with incidental repeats — normal transcription — must
        // decode on untouched.
        var tokens = new List<int> { StartOfTranscriptToken, TranscribeToken, NoTimestampsToken };
        tokens.AddRange([TheLower, 1002, 634, TheLower, 415, 5663, 13, TheLower, 293, 19882, 1057, 17913, 343]);

        WhisperDecoder.TryDetectRepetitionCycle(tokens, GeneratedStart, out _)
            .Should().BeFalse();
    }

    [Fact]
    public void TryDetectRepetitionCycle_ShortGeneratedTail_DoesNotFire()
    {
        // Nothing to judge yet: fewer generated tokens than the minimum span can never qualify,
        // and the check must not read back into the prompt prefix looking for more.
        var tokens = new List<int>
        {
            StartOfTranscriptToken, TranscribeToken, NoTimestampsToken, TheLower, TheLower
        };

        WhisperDecoder.TryDetectRepetitionCycle(tokens, GeneratedStart, out _)
            .Should().BeFalse();
    }
}
