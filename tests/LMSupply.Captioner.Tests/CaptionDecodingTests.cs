using AwesomeAssertions;
using LMSupply.Captioner.Inference;

namespace LMSupply.Captioner.Tests;

/// <summary>
/// <see cref="CaptionerOptions.NumBeams"/> is a beam search and <see cref="CaptionerOptions.Temperature"/> a sampling
/// temperature — before 0.101.0 the first meant "sample" and the second was read only when the first was above 1.
/// </summary>
public sealed class CaptionDecodingTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private const int End = 0;

    /// <summary>
    /// A toy decoder over tokens {0 = end, 1, 2, 3}. Greedy takes token 1 first (0.5 against 0.4), after which every
    /// continuation is unlikely; token 2 first leads to a near-certain 3 and end. The best whole sequence is 2 3, which
    /// only a beam search finds.
    /// </summary>
    private static Task<(float[] Logits, object? State)> ToyStep(IReadOnlyList<int> sequence, object? state)
    {
        float[] probs = sequence.Count switch
        {
            1 => [0.0f, 0.5f, 0.4f, 0.1f],
            _ when sequence[^1] == 1 => [0.3f, 0.25f, 0.2f, 0.25f],
            _ when sequence[^1] == 2 => [0.01f, 0.01f, 0.01f, 0.97f],
            _ => [0.97f, 0.01f, 0.01f, 0.01f]
        };
        return Task.FromResult((probs.Select(p => MathF.Log(Math.Max(p, 1e-9f))).ToArray(), state));
    }

    [Fact]
    public async Task BeamSearch_FindsTheSequenceGreedyMisses()
    {
        var hypotheses = await BeamSearch<object?>.RunAsync([9], null, beams: 2, maxNewTokens: 5, End, ToyStep, null, Ct);

        hypotheses[0].Tokens.Should().Equal(2, 3);
        hypotheses.Should().HaveCountGreaterThan(1, "the runner-up beams are kept as alternatives");
    }

    [Fact]
    public async Task Greedy_TakesTheLocallyBestToken()
    {
        // The control: the same toy decoder decoded greedily starts with 1.
        var (first, _) = NextToken.Choose((await ToyStep([9], null)).Logits, new CaptionerOptions());

        first.Should().Be(1);
    }

    [Fact]
    public async Task BeamSearch_StopsAtTheStepBudget_AndStillAnswers()
    {
        // Never emits the end token.
        static Task<(float[], object?)> Loop(IReadOnlyList<int> s, object? state)
            => Task.FromResult((new[] { float.NegativeInfinity, 0f, -1f, -2f }, state));

        var hypotheses = await BeamSearch<object?>.RunAsync([9], null, beams: 2, maxNewTokens: 3, End, Loop, null, Ct);

        hypotheses[0].Tokens.Should().HaveCount(3);
    }

    [Theory]
    [InlineData(0, null)]
    [InlineData(1, 0f)]
    [InlineData(3, 0.7f)]
    public async Task ContradictoryOrInvalidDecodingOptions_AreRefusedBeforeAnyDownload(int beams, float? temperature)
    {
        var options = new CaptionerOptions { NumBeams = beams, Temperature = temperature, DisableAutoDownload = true };

        var load = () => LocalCaptioner.LoadAsync("default", options, cancellationToken: Ct);

        await load.Should().ThrowAsync<NotSupportedException>();
    }
}
