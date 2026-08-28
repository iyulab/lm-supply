using AwesomeAssertions;
using LMSupply.Transcriber;
using LMSupply.Transcriber.Decoding;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// Tests for <see cref="WhisperDecoder.FinalizeSegments"/> — the post-decode-loop logic that
/// closes out any segment that never received a timestamp token. Pure token/text logic, no ONNX
/// session involved, so real audio is not required to cover it.
/// </summary>
public class WhisperDecoderSegmentFinalizationTests
{
    // Minimal single-entry vocab: id 1000 decodes to "a" (byte 'a' is its own byte-to-unicode
    // mapping, so this round-trips through Decode() without needing a real model's vocab.json).
    private const int TextTokenId = 1000;

    private static WhisperTokenizer CreateTokenizer() =>
        WhisperTokenizer.CreateForTesting(
            new Dictionary<int, string> { [TextTokenId] = "a" },
            new Dictionary<string, int> { ["a"] = TextTokenId });

    [Fact]
    public void FinalizeSegments_NoTimestampTokenGenerated_UsesChunkDurationNotHardcoded30Seconds()
    {
        // Reproduces the reported failure shape: the decoder emits a few plain-text tokens and
        // then hits EOT before any timestamp token is ever generated (see docket 9e8688d2's
        // 2026-08-23 trace: SOT + "The" x3 + EOT, zero timestamp tokens). In that case the only
        // open segment is `currentSegmentTokens`, which the "remaining tokens as final segment"
        // branch must close using the chunk's real duration — not a fixed 30s.
        var tokenizer = CreateTokenizer();
        var decoder = WhisperDecoder.CreateForTesting(tokenizer);

        var segments = new List<TranscriptionSegment>();
        var initialTokens = tokenizer.GetSotSequence(language: "en", timestamps: false, translate: false);
        var tokens = new List<int>(initialTokens) { TextTokenId };
        var currentSegmentTokens = new List<int> { TextTokenId };
        var currentSegmentLogProbs = new List<float> { -0.1f };

        const double actualChunkDurationSeconds = 14.6;

        decoder.FinalizeSegments(
            segments,
            tokens,
            initialTokens,
            currentSegmentTokens,
            currentSegmentLogProbs,
            segmentId: 0,
            currentSegmentStart: 0.0,
            chunkNoSpeechProb: 3.08e-05f,
            chunkDurationSeconds: actualChunkDurationSeconds);

        segments.Should().ContainSingle();
        segments[0].End.Should().Be(actualChunkDurationSeconds,
            "the segment's end time must reflect the chunk's real pre-padding duration, " +
            "not the fixed 30s window the encoder always pads/truncates to");
        segments[0].End.Should().NotBe(30.0);
    }

    [Fact]
    public void FinalizeSegments_NoSegmentsAndNoTimestampTokens_SingleSegmentUsesChunkDuration()
    {
        // No-timestamps mode (or EOT on the very first step): segments is empty going in, so the
        // "no segments created" branch builds one segment from the whole token stream.
        var tokenizer = CreateTokenizer();
        var decoder = WhisperDecoder.CreateForTesting(tokenizer);

        var segments = new List<TranscriptionSegment>();
        var initialTokens = tokenizer.GetSotSequence(language: "en", timestamps: false, translate: false);
        var tokens = new List<int>(initialTokens) { TextTokenId };

        const double actualChunkDurationSeconds = 20.0;

        decoder.FinalizeSegments(
            segments,
            tokens,
            initialTokens,
            currentSegmentTokens: [],
            currentSegmentLogProbs: [],
            segmentId: 0,
            currentSegmentStart: 0.0,
            chunkNoSpeechProb: null,
            chunkDurationSeconds: actualChunkDurationSeconds);

        segments.Should().ContainSingle();
        segments[0].End.Should().Be(actualChunkDurationSeconds);
    }
}
