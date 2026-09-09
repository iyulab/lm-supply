namespace LMSupply.Transcriber.Decoding;

/// <summary>
/// 한 프레임에 대한 prediction-network + joint 실행 결과.
/// <see cref="Logits"/>는 토큰 logits(어휘 크기) 뒤에 지속시간(duration) bin logits가 이어진 joint 출력 그대로다.
/// </summary>
internal readonly record struct TdtJointOutput(float[] Logits, float[] StateH, float[] StateC);

/// <summary>
/// 프레임 <paramref name="frameIndex"/>의 인코더 출력과 마지막 토큰·LSTM 상태로 joint를 한 번 돌린다.
/// 디코더 ONNX 세션을 감싼 델리게이트라 순수 함수인 <see cref="TdtGreedyDecoder"/>를 가짜로도 검증할 수 있다.
/// </summary>
internal delegate TdtJointOutput TdtJointStep(int frameIndex, int lastToken, float[] stateH, float[] stateC);

/// <summary>디코딩된 토큰 하나 — 어느 인코더 프레임에서 나왔는지와 그 확률.</summary>
internal readonly record struct TdtToken(int Id, int Frame, float LogProb);

/// <summary>
/// Token-and-Duration Transducer의 greedy 디코딩(onnx-asr `asr.py` 규칙 그대로).
/// 프레임마다 (마지막 토큰, 상태)로 joint를 돌려 토큰과 «몇 프레임을 건너뛸지»를 함께 읽는다.
/// blank가 아닐 때만 토큰을 내고 상태를 갱신한다; duration이 0이면 같은 프레임에서 계속 내되
/// <see cref="MaxSymbolsPerStep"/>에서 강제로 한 프레임 전진한다 — whisper식 runaway decode가 구조적으로 없는 이유다.
/// </summary>
internal static class TdtGreedyDecoder
{
    /// <summary>같은 프레임에서 연속으로 낼 수 있는 최대 심볼 수(NeMo `max_symbols`, onnx-asr 기본 10).</summary>
    public const int MaxSymbolsPerStep = 10;

    /// <param name="encodedLength">유효 인코더 프레임 수.</param>
    /// <param name="vocabSize">토큰 logits 길이(blank 포함). 그 뒤의 logits는 duration bin이다.</param>
    /// <param name="blankId">blank 토큰 id.</param>
    /// <param name="stateSize">LSTM h/c 각각의 원소 수(layers × hidden).</param>
    /// <param name="step">joint 실행.</param>
    /// <param name="maxSymbolsPerStep">같은 프레임에서 연속으로 낼 수 있는 최대 심볼 수 — 이 수에 닿으면 강제로 한 프레임 전진한다.</param>
    public static List<TdtToken> Decode(
        int encodedLength,
        int vocabSize,
        int blankId,
        int stateSize,
        TdtJointStep step,
        int maxSymbolsPerStep = MaxSymbolsPerStep)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(encodedLength);
        ArgumentOutOfRangeException.ThrowIfLessThan(vocabSize, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxSymbolsPerStep, 1);

        var tokens = new List<TdtToken>();
        var stateH = new float[stateSize];
        var stateC = new float[stateSize];
        var t = 0;
        var emittedAtFrame = 0;

        while (t < encodedLength)
        {
            var last = tokens.Count > 0 ? tokens[^1].Id : blankId;
            var joint = step(t, last, stateH, stateC);
            var logits = joint.Logits;
            if (logits.Length <= vocabSize)
                throw new InvalidDataException($"Joint output has {logits.Length} logits but at least {vocabSize + 1} were expected (tokens + duration bins).");

            var token = ArgMax(logits, 0, vocabSize);
            var duration = ArgMax(logits, vocabSize, logits.Length) - vocabSize;

            if (token != blankId)
            {
                stateH = joint.StateH;
                stateC = joint.StateC;
                tokens.Add(new TdtToken(token, t, LogSoftmaxAt(logits, vocabSize, token)));
                emittedAtFrame++;
            }

            if (duration > 0)
            {
                t += duration;
                emittedAtFrame = 0;
            }
            else if (token == blankId || emittedAtFrame >= maxSymbolsPerStep)
            {
                t += 1;
                emittedAtFrame = 0;
            }
        }

        return tokens;
    }

    private static int ArgMax(float[] values, int start, int end)
    {
        var best = start;
        for (var i = start + 1; i < end; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }

    private static float LogSoftmaxAt(float[] logits, int count, int index)
    {
        var max = float.NegativeInfinity;
        for (var i = 0; i < count; i++) max = Math.Max(max, logits[i]);
        double sum = 0;
        for (var i = 0; i < count; i++) sum += Math.Exp(logits[i] - max);
        return (float)(logits[index] - max - Math.Log(sum));
    }
}
