using AwesomeAssertions;
using LMSupply.Transcriber.Decoding;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// Tests for <see cref="WhisperDecoder.SelectNextToken"/> — the per-decode-step selection logic
/// (hallucination-suppression guard, greedy argmax). Pure logit-array logic, no ONNX session
/// involved.
/// </summary>
public class WhisperDecoderRepeatSuppressionTests
{
    // Real production special-token ids (WhisperTokenizer's v2 defaults) and a large-enough vocab
    // so the exact indices used in the captured trace below are addressable.
    private const int EndOfTextToken = 50257;
    private const int StartOfTranscriptToken = 50258;
    private const int TranscribeToken = 50359;
    private const int NoTimestampsToken = 50363;
    private const int RepeatedToken = 440; // "The"
    private const int ContinuationToken = 264;
    private const int OtherCandidateToken = 13;
    private const int VocabSize = 51865;

    private static float[] CreateBaselineLogits()
    {
        var logits = new float[VocabSize];
        Array.Fill(logits, -1000f);
        return logits;
    }

    [Fact]
    public void SelectNextToken_EotWinsByNarrowMarginRightAfterSuppression_PicksRealContinuationInstead()
    {
        // Reproduces the exact decode-step state captured in docket iyulab/lm-supply#59's
        // 2026-08-30 trace (LMSupply.Transcriber 0.42.4, real fixture, CPU EP): SOT sequence
        // [50258, 50359], then NoTimestamps(50363) + "The"x3, so at step 4 the repeat-suppression
        // guard fires on token 440. The reported post-guard top-5 was
        // [50257:2.699 (EOT), 264:2.534, 13:0.404, ...] — EOT wins the unpatched decoder by a
        // margin of just 0.165 over the real continuation candidate (264), because the guard only
        // ever touched the repeated token's logit and never EOT's.
        var decoder = WhisperDecoder.CreateForTesting(WhisperTokenizer.CreateDefault());
        var initialTokens = new[] { StartOfTranscriptToken, TranscribeToken };
        var tokens = new List<int>(initialTokens) { NoTimestampsToken, RepeatedToken, RepeatedToken, RepeatedToken };

        var logits = CreateBaselineLogits();
        logits[EndOfTextToken] = 2.699f;
        logits[ContinuationToken] = 2.534f;
        logits[OtherCandidateToken] = 0.404f;
        logits[RepeatedToken] = 5.0f;         // overwritten to -Inf by the guard regardless

        var selected = decoder.SelectNextToken(logits, tokens, initialTokens, options: null);

        selected.Should().Be(ContinuationToken,
            "a suppression event should cost EOT a soft penalty, so it can no longer win by a " +
            "margin (0.165 in the captured trace) smaller than that penalty");
        selected.Should().NotBe(EndOfTextToken);
    }

    [Fact]
    public void SelectNextToken_NoSuppressionEvent_EotStillWinsNormally()
    {
        // Guards against over-correcting: when the last three tokens are NOT identical, the
        // hallucination guard never fires, so EOT must remain untouched and free to win a
        // genuine early stop exactly as before this fix.
        var decoder = WhisperDecoder.CreateForTesting(WhisperTokenizer.CreateDefault());
        var initialTokens = new[] { StartOfTranscriptToken, TranscribeToken };
        var tokens = new List<int>(initialTokens) { NoTimestampsToken, 111, 222, 333 };

        var logits = CreateBaselineLogits();
        logits[EndOfTextToken] = 5.0f;
        logits[ContinuationToken] = 1.0f;

        var selected = decoder.SelectNextToken(logits, tokens, initialTokens, options: null);

        selected.Should().Be(EndOfTextToken);
    }

    [Fact]
    public void SelectNextToken_SuppressionEvent_RepeatedTokenNeverWins()
    {
        // The pre-existing half of the guard must still hold: the repeated token itself is never
        // selectable again immediately after a suppression event, even if nothing else scores well.
        var decoder = WhisperDecoder.CreateForTesting(WhisperTokenizer.CreateDefault());
        var initialTokens = new[] { StartOfTranscriptToken, TranscribeToken };
        var tokens = new List<int>(initialTokens) { NoTimestampsToken, RepeatedToken, RepeatedToken, RepeatedToken };

        var logits = CreateBaselineLogits();
        logits[RepeatedToken] = 100f; // would trivially win an unguarded argmax
        logits[ContinuationToken] = -5f;
        logits[EndOfTextToken] = -10f;

        var selected = decoder.SelectNextToken(logits, tokens, initialTokens, options: null);

        selected.Should().NotBe(RepeatedToken);
        selected.Should().Be(ContinuationToken);
    }

    [Fact]
    public void SelectNextToken_RecentlyUsedSentenceOpener_IsNotTaxedIntoLosingToEot()
    {
        // Docket iyulab/lm-supply#253: "... about the delayed order. The next meeting is ..." —
        // the opener of the final sentence ("The") had already appeared a few tokens earlier. A
        // blanket 1.2 penalty on every token of the last ten dropped its logit from 3.0 to 2.5, so
        // end-of-text (2.8) won and the clip's final sentence was never decoded.
        var decoder = WhisperDecoder.CreateForTesting(WhisperTokenizer.CreateDefault());
        var initialTokens = new[] { StartOfTranscriptToken, TranscribeToken };
        var tokens = new List<int>(initialTokens) { NoTimestampsToken, RepeatedToken, 111, 222, 333, OtherCandidateToken };

        var logits = CreateBaselineLogits();
        logits[RepeatedToken] = 3.0f;
        logits[EndOfTextToken] = 2.8f;

        var selected = decoder.SelectNextToken(logits, tokens, initialTokens, options: null);

        selected.Should().Be(RepeatedToken, "a token that already occurred is still a legitimate continuation");
    }

    [Fact]
    public void SelectNextToken_NoSuppressionEvent_LeavesRecentTokenLogitsUntouched()
    {
        // Outside a suppression event nothing reweights recently used tokens — neither a positive
        // logit (divided, before #253) nor a negative one (multiplied).
        var decoder = WhisperDecoder.CreateForTesting(WhisperTokenizer.CreateDefault());
        var initialTokens = new[] { StartOfTranscriptToken, TranscribeToken };
        var tokens = new List<int>(initialTokens) { NoTimestampsToken, RepeatedToken, OtherCandidateToken };

        var logits = CreateBaselineLogits();
        logits[RepeatedToken] = 3.0f;
        logits[OtherCandidateToken] = -2.0f;

        decoder.SelectNextToken(logits, tokens, initialTokens, options: null);

        logits[RepeatedToken].Should().Be(3.0f);
        logits[OtherCandidateToken].Should().Be(-2.0f);
    }
}
