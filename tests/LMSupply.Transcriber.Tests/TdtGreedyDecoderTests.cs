using AwesomeAssertions;
using LMSupply.Transcriber.Decoding;

namespace LMSupply.Transcriber.Tests;

/// <summary>
/// Pins the Token-and-Duration Transducer greedy loop against a scripted joint: which frame advances, when a token is
/// emitted, when the prediction-network state moves, and the per-frame symbol cap that makes runaway output impossible.
/// Vocabulary: ids 0..2 are tokens, 3 is blank; five duration bins (0..4) like the real export.
/// </summary>
public sealed class TdtGreedyDecoderTests
{
    private const int Vocab = 4;
    private const int Blank = 3;
    private const int StateSize = 2;

    /// <summary>Builds joint logits: <paramref name="token"/> wins the token block, <paramref name="duration"/> wins the duration block.</summary>
    private static float[] Logits(int token, int duration, int durationBins = 5)
    {
        var logits = new float[Vocab + durationBins];
        logits[token] = 5f;
        logits[Vocab + duration] = 5f;
        return logits;
    }

    private sealed class ScriptedJoint
    {
        private readonly Queue<(int Token, int Duration)> _script;
        public List<(int Frame, int LastToken, float StateMarker)> Calls { get; } = [];
        private int _callNo;

        public ScriptedJoint(params (int Token, int Duration)[] script) => _script = new Queue<(int, int)>(script);

        public TdtJointOutput Step(int frame, int lastToken, float[] stateH, float[] stateC)
        {
            Calls.Add((frame, lastToken, stateH[0]));
            var (token, duration) = _script.Count > 0 ? _script.Dequeue() : (Blank, 1);
            _callNo++;
            // Each call returns a distinct state so the test can see whether the decoder kept it.
            return new TdtJointOutput(Logits(token, duration), [_callNo, 0], [_callNo, 0]);
        }
    }

    [Fact]
    public void Blank_WithZeroDuration_AdvancesOneFrame_EmitsNothing_KeepsState()
    {
        var joint = new ScriptedJoint((Blank, 0), (Blank, 0));

        var tokens = TdtGreedyDecoder.Decode(encodedLength: 2, Vocab, Blank, StateSize, joint.Step);

        tokens.Should().BeEmpty();
        joint.Calls.Select(c => c.Frame).Should().Equal(0, 1);
        joint.Calls.Select(c => c.StateMarker).Should().Equal([0f, 0f], "a blank must not advance the prediction-network state");
        joint.Calls.Select(c => c.LastToken).Should().Equal(Blank, Blank);
    }

    [Fact]
    public void Token_WithZeroDuration_StaysOnFrame_EmitsAndAdvancesState()
    {
        var joint = new ScriptedJoint((1, 0), (2, 0), (Blank, 0));

        var tokens = TdtGreedyDecoder.Decode(encodedLength: 1, Vocab, Blank, StateSize, joint.Step);

        tokens.Select(t => t.Id).Should().Equal(1, 2);
        tokens.Select(t => t.Frame).Should().Equal(0, 0);
        joint.Calls.Select(c => c.Frame).Should().Equal(0, 0, 0);
        joint.Calls.Select(c => c.LastToken).Should().Equal([Blank, 1, 2], "the decoder feeds back the last emitted token");
        joint.Calls.Select(c => c.StateMarker).Should().Equal([0f, 1f, 2f], "state advances after each non-blank");
        tokens.Should().OnlyContain(t => t.LogProb < 0f && t.LogProb > -1f);
    }

    [Fact]
    public void Duration_SkipsThatManyFrames_EvenAfterAToken()
    {
        var joint = new ScriptedJoint((1, 3), (2, 1));

        var tokens = TdtGreedyDecoder.Decode(encodedLength: 5, Vocab, Blank, StateSize, (f, l, h, c) =>
        {
            var o = joint.Step(f, l, h, c);
            return o;
        });

        // frame 0: token 1, skip 3 → frame 3: token 2, skip 1 → frame 4: script exhausted → blank, +1 → done
        tokens.Select(t => (t.Id, t.Frame)).Should().Equal((1, 0), (2, 3));
        joint.Calls.Select(c => c.Frame).Should().Equal(0, 3, 4);
    }

    [Fact]
    public void MaxSymbolsPerStep_ForcesFrameAdvance_SoAStuckJointCannotLoopForever()
    {
        var alwaysToken = new ScriptedJoint(Enumerable.Repeat((1, 0), 100).ToArray());

        var tokens = TdtGreedyDecoder.Decode(encodedLength: 2, Vocab, Blank, StateSize, alwaysToken.Step, maxSymbolsPerStep: 3);

        tokens.Should().HaveCount(6, "3 symbols per frame × 2 frames, then the loop ends");
        tokens.Select(t => t.Frame).Should().Equal(0, 0, 0, 1, 1, 1);
    }

    [Fact]
    public void EmptyInput_ReturnsNoTokens_AndNeverCallsTheJoint()
    {
        var joint = new ScriptedJoint();

        TdtGreedyDecoder.Decode(encodedLength: 0, Vocab, Blank, StateSize, joint.Step).Should().BeEmpty();

        joint.Calls.Should().BeEmpty();
    }

    [Fact]
    public void JointWithoutDurationBins_IsRejected()
    {
        var act = () => TdtGreedyDecoder.Decode(1, Vocab, Blank, StateSize, (_, _, _, _) => new TdtJointOutput(new float[Vocab], new float[StateSize], new float[StateSize]));

        act.Should().Throw<InvalidDataException>().WithMessage("*duration bins*");
    }
}
