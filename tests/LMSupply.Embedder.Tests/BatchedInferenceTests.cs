using AwesomeAssertions;
using Xunit;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// A batch of texts is embedded in batched inference runs and gives the same vectors as embedding each text alone.
/// Before 0.121.0 the batch was padded to its longest text and then run one text at a time, so every text paid the
/// batch's padding and nothing ran together (measured: 64 passages per call were slower than one per call). Needs the
/// model download; Integration keeps it out of CI.
/// </summary>
[Trait("Category", "Integration")]
public sealed class BatchedInferenceTests
{
    [Fact]
    public async Task ABatch_GivesTheSameVectors_AsEmbeddingEachTextAlone()
    {
        var ct = TestContext.Current.CancellationToken;
        await using var model = await LocalEmbedder.LoadAsync("default", new EmbedderOptions { Provider = ExecutionProvider.Cpu },
            cancellationToken: ct);

        // Forty texts from one word to past the model's 512-token limit, out of order: their padded total exceeds one
        // run's budget, so the batch is split into several runs and padded within each.
        var texts = Enumerable.Range(0, 40)
            .Select(i => (i * 17) % 40)
            .Select(n => n == 0 ? "hello" : string.Join(' ', Enumerable.Repeat("The keyword index stores terms analyzed at write time.", n)))
            .ToArray();

        var batch = await model.EmbedAsync(texts, ct);

        for (var i = 0; i < texts.Length; i++)
        {
            var alone = await model.EmbedAsync(texts[i], ct);
            var cosine = LocalEmbedder.CosineSimilarity(batch[i], alone);
            cosine.Should().BeGreaterThan(0.9999f, $"text {i} must embed the same in a batch as alone");
        }
    }

    [Fact]
    public async Task AnEmptyBatch_ReturnsNothing()
    {
        var ct = TestContext.Current.CancellationToken;
        await using var model = await LocalEmbedder.LoadAsync("default", new EmbedderOptions { Provider = ExecutionProvider.Cpu },
            cancellationToken: ct);

        (await model.EmbedAsync(Array.Empty<string>(), ct)).Should().BeEmpty();
    }
}

/// <summary>How a batch is split into inference runs — pure, no model.</summary>
public sealed class BatchRunPlanTests
{
    [Theory]
    [InlineData(new[] { 10, 500, 20, 30 }, 512, "0,2,3|1")]        // 3 x 30 = 90 fits; adding 500 would make 4 x 500
    [InlineData(new[] { 100, 100, 100 }, 250, "0,1|2")]              // two of 100 fit in 250, a third does not
    [InlineData(new[] { 900, 5 }, 512, "1|0")]                       // a text longer than the budget runs alone
    [InlineData(new[] { 7 }, 8192, "0")]
    public void PlanRuns_GroupsByLength_WithinTheBudget(int[] lengths, int budget, string expected)
    {
        var plan = string.Join('|', EmbeddingModel.PlanRuns(lengths, budget).Select(run => string.Join(',', run)));

        plan.Should().Be(expected);
    }
}
