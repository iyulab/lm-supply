using AwesomeAssertions;
using LMSupply.Transcriber.Internal;
using Xunit;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// A segment whose entire content is one word repeated is not a transcription, and the
/// compression-ratio filter cannot say so: deflate needs length before repetition shows up as a
/// ratio, and these segments are two or three words long.
/// <para>
/// Both shapes below come from a consumer's decode-step trace against a real fixture. The
/// repetition guard ends a chunk once the decoder settles into a cycle, but the text it keeps from
/// before the cycle is itself the degeneration — and a chunk that reaches end-of-text after only
/// three identical tokens never becomes a cycle at all, so nothing upstream of here sees either.
/// </para>
/// </summary>
public class SegmentPostProcessorDegenerateRepetitionTests
{
    [Fact]
    public void Process_SegmentThatIsOneWordRepeatedThreeTimes_IsDropped()
    {
        // Decoder emitted the same token three times, then end-of-text: below any cycle length,
        // and far too short for the compression-ratio filter to reject.
        var segments = new List<TranscriptionSegment> { Segment("The The The") };

        var (result, text) = SegmentPostProcessor.Process(segments, options: null);

        result.Should().BeEmpty();
        text.Should().BeEmpty();
    }

    [Fact]
    public void Process_TextKeptFromBeforeADetectedCycle_IsDropped()
    {
        // The cycle detector trimmed the cyclic tail and kept what preceded it — but the prefix is
        // the same word twice, which is what the cycle was made of.
        var segments = new List<TranscriptionSegment> { Segment("The The") };

        var (result, _) = SegmentPostProcessor.Process(segments, options: null);

        result.Should().BeEmpty();
    }

    [Fact]
    public void Process_LongerSegmentAlternatingTwoWords_IsDropped()
    {
        // The same failure one rung out: a period-2 loop whose kept prefix is long enough to look
        // like text but carries only two distinct words.
        var segments = new List<TranscriptionSegment> { Segment("the The the The the The") };

        var (result, _) = SegmentPostProcessor.Process(segments, options: null);

        result.Should().BeEmpty();
    }

    [Fact]
    public void Process_LongerSegmentAlternatingTwoDistinctWords_IsDropped()
    {
        // Two genuinely different words alternating -- the same period-2 shape as above, but
        // without the capitalisation that collapses to a single distinct word.
        var segments = new List<TranscriptionSegment> { Segment("the and the and the and") };

        var (result, _) = SegmentPostProcessor.Process(segments, options: null);

        result.Should().BeEmpty();
    }

    /// <summary>
    /// The length floor on the two-distinct-word rule: four words built from two is short enough to
    /// be a real utterance, so the check must not reach down into ordinary short speech.
    /// </summary>
    [Fact]
    public void Process_ShortSegmentWithTwoDistinctWords_IsKept()
    {
        var segments = new List<TranscriptionSegment> { Segment("the cat the cat") };

        var (result, _) = SegmentPostProcessor.Process(segments, options: null);

        result.Should().ContainSingle();
    }

    /// <summary>
    /// A segment that is one word repeated is dropped however long it runs -- there is no length at
    /// which "one word and nothing else" becomes a transcription, and any ceiling that spared this
    /// would equally spare the three-word case this check exists for.
    /// </summary>
    /// <remarks>
    /// The accepted cost: audio where someone really does say only "no no no" loses that segment.
    /// A caller who needs those should raise it -- the alternative is a filter that cannot state a
    /// consistent rule.
    /// </remarks>
    [Fact]
    public void Process_OneWordRepeatedManyTimes_IsAlsoDropped()
    {
        var segments = new List<TranscriptionSegment> { Segment("no no no no no") };

        var (result, _) = SegmentPostProcessor.Process(segments, options: null);

        result.Should().BeEmpty();
    }

    /// <summary>
    /// The known Whisper silence hallucination is a single word, not a repetition, and a consumer
    /// has explicitly scoped it out of this defect. Dropping it here would be a behaviour change
    /// beyond what the evidence supports.
    /// </summary>
    [Fact]
    public void Process_SingleWordSegment_IsKept()
    {
        var segments = new List<TranscriptionSegment> { Segment("you") };

        var (result, _) = SegmentPostProcessor.Process(segments, options: null);

        result.Should().ContainSingle().Which.Text.Should().Be("you");
    }

    /// <summary>
    /// A repeated word inside real speech must survive: the check looks at what a segment is made
    /// of as a whole, not at whether any word occurs twice.
    /// </summary>
    [Theory]
    [InlineData("very very good")]
    [InlineData("The the quick brown fox")]
    [InlineData("no no I meant the other one")]
    public void Process_RealSpeechContainingARepeatedWord_IsKept(string spoken)
    {
        var segments = new List<TranscriptionSegment> { Segment(spoken) };

        var (result, _) = SegmentPostProcessor.Process(segments, options: null);

        result.Should().ContainSingle().Which.Text.Should().Be(spoken);
    }

    /// <summary>
    /// Whisper's own special-token text is a single bracketed marker, not a repetition — and a
    /// consumer verified it as legitimate output for genuinely silent audio.
    /// </summary>
    [Fact]
    public void Process_BlankAudioMarker_IsKept()
    {
        var segments = new List<TranscriptionSegment> { Segment("[BLANK_AUDIO]") };

        var (result, _) = SegmentPostProcessor.Process(segments, options: null);

        result.Should().ContainSingle();
    }

    /// <summary>
    /// Dropping a degenerate segment must not take its neighbours with it: the consumer's fixture
    /// produced one degenerate chunk beside a legitimate one.
    /// </summary>
    [Fact]
    public void Process_DegenerateSegmentBesideARealOne_DropsOnlyTheDegenerateSegment()
    {
        var segments = new List<TranscriptionSegment>
        {
            Segment("The The The", start: 30, end: 60),
            Segment("[BLANK_AUDIO]", start: 60, end: 70)
        };

        var (result, _) = SegmentPostProcessor.Process(segments, options: null);

        result.Should().ContainSingle().Which.Text.Should().Be("[BLANK_AUDIO]");
        result[0].Id.Should().Be(0, "surviving segments are re-indexed");
    }

    private static TranscriptionSegment Segment(string text, double start = 0, double end = 30) => new()
    {
        Id = 0,
        Start = start,
        End = end,
        Text = text,
        // Deliberately set: these texts are short enough that the compression-ratio filter passes
        // them, which is the whole reason this check has to exist separately.
        CompressionRatio = SegmentPostProcessor.ComputeCompressionRatio(text)
    };
}
