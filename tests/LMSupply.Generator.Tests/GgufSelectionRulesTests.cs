using AwesomeAssertions;
using LMSupply.Generator.Gguf;
using Xunit;

namespace LMSupply.Generator.Tests;

/// <summary>
/// The automatic selection rule — the largest model that fits, the KV cache counted, the system-RAM path, the fallback —
/// over made-up pools. These cases do not name a model the registry ships, so replacing a shipped model changes none of
/// them; what the shipped pool picks on a given card is stated, with its numbers, in <see cref="GgufModelRegistryTests"/>.
/// </summary>
public class GgufSelectionRulesTests
{
    private const long GiB = 1024L * 1024 * 1024;
    private const long MiB = 1024L * 1024;
    private const int DefaultContext = 4096;

    private static KeyValuePair<string, GgufModelInfo> Entry(string alias, long weightsBytes, long kvBytesPerToken, params string[] knownIssues) =>
        new(alias, new GgufModelInfo
        {
            RepoId = $"example/{alias}",
            DisplayName = alias,
            DefaultFile = $"{alias}.gguf",
            ChatFormat = "chatml",
            ContextLength = 1_048_576,
            ParameterCount = weightsBytes / 2,
            EstimatedSizeBytes = weightsBytes,
            QuantizationType = "Q4_K_M",
            KvCacheBytesPerToken = kvBytesPerToken,
            License = LicenseTier.MIT,
            KnownIssues = knownIssues,
        });

    private static GgufModelInfo Fallback() => Entry("fallback", GiB, 16 * 1024).Value;

    private static ModelSelectionResult Select(
        IEnumerable<KeyValuePair<string, GgufModelInfo>> pool,
        long vramBytes,
        long systemRamBytes = 0,
        int context = DefaultContext,
        IReadOnlyCollection<string>? excludeKnownIssues = null,
        AutoSelectionGoal goal = AutoSelectionGoal.Quality) =>
        GgufModelRegistry.Select(pool, Fallback, vramBytes, safetyMargin: 0, systemRamBytes, context, excludeKnownIssues, goal);

    [Fact]
    public void TheLargestModelThatFitsTheVramIsSelected()
    {
        KeyValuePair<string, GgufModelInfo>[] pool =
            [Entry("tiny", 1 * GiB, 16 * 1024), Entry("mid", 4 * GiB, 32 * 1024), Entry("big", 12 * GiB, 8 * 1024), Entry("huge", 40 * GiB, 64 * 1024)];

        var result = Select(pool, vramBytes: 20 * GiB);

        result.Selected.AliasName.Should().Be("big");
        result.Reason.Should().Be(ModelSelectionReason.Fits);
        result.Candidates.Select(c => c.Model.AliasName).Should().Equal("huge", "big", "mid", "tiny");
        result.Candidates.Where(c => c.Fits).Select(c => c.Model.AliasName).Should().Equal("big", "mid", "tiny");
    }

    [Fact]
    public void TheKvCacheCountsTowardsWhetherAModelFits()
    {
        // "cache-heavy" holds 6 GiB of weights: the weights alone fit 8 GiB, but its 1 MiB per token adds 4 GiB at 4,096 tokens.
        KeyValuePair<string, GgufModelInfo>[] pool = [Entry("cache-heavy", 6 * GiB, 1 * MiB), Entry("lean", 3 * GiB, 16 * 1024)];

        var result = Select(pool, vramBytes: 8 * GiB);

        result.Selected.AliasName.Should().Be("lean");
        result.Reason.Should().Be(ModelSelectionReason.Fits);
    }

    [Fact]
    public void ALongerContextMovesTheChoiceToTheModelWhoseCacheIsSmaller()
    {
        // Same 12 GiB, same two models. At 4,096 tokens the larger-cache one fits and holds more, so it wins; at 32,768 tokens its
        // cache (8 GiB) no longer fits beside its weights, and the lean-cache model takes over.
        KeyValuePair<string, GgufModelInfo>[] pool = [Entry("weights6-cache-heavy", 6 * GiB, 256 * 1024), Entry("weights4-cache-lean", 4 * GiB, 8 * 1024)];

        Select(pool, vramBytes: 12 * GiB, context: 4_096).Selected.AliasName.Should().Be("weights6-cache-heavy");
        Select(pool, vramBytes: 12 * GiB, context: 32_768).Selected.AliasName.Should().Be("weights4-cache-lean");
    }

    [Fact]
    public void WhenNothingFitsTheVramTheSystemRamPathPicksByGoal()
    {
        var ram = 64 * GiB;
        var budget = GgufModelRegistry.SystemRamBudgetBytes(ram);
        // Sizes relative to the budget so the cases hold whatever the RAM share is.
        KeyValuePair<string, GgufModelInfo>[] pool =
            [Entry("small", budget / 8, 16 * 1024), Entry("medium", budget / 3, 16 * 1024), Entry("large", budget / 2, 16 * 1024), Entry("too-large", budget * 2, 16 * 1024)];

        var quality = Select(pool, vramBytes: 0, systemRamBytes: ram, goal: AutoSelectionGoal.Quality);
        var responsive = Select(pool, vramBytes: 0, systemRamBytes: ram, goal: AutoSelectionGoal.Responsive);
        var quickStart = Select(pool, vramBytes: 0, systemRamBytes: ram, goal: AutoSelectionGoal.QuickStart);

        quality.Reason.Should().Be(ModelSelectionReason.FitsInSystemRam);
        quality.Selected.AliasName.Should().Be("large", because: "Quality takes the largest model the RAM budget holds");
        responsive.Selected.AliasName.Should().Be("small", because: "Responsive takes the smallest model of the pool that the RAM holds");
        quickStart.Selected.AliasName.Should().Be("small", because: "QuickStart weighs the download, the smallest weights file");
        new[] { quality, responsive, quickStart }.Should().AllSatisfy(r => r.Reason.Should().Be(ModelSelectionReason.FitsInSystemRam));
    }

    [Fact]
    public void WhenAModelFitsTheVramTheSystemRamIsNotConsulted()
    {
        KeyValuePair<string, GgufModelInfo>[] pool = [Entry("small", 2 * GiB, 16 * 1024), Entry("large", 10 * GiB, 16 * 1024)];

        var result = Select(pool, vramBytes: 6 * GiB, systemRamBytes: 256 * GiB);

        result.Selected.AliasName.Should().Be("small");
        result.Reason.Should().Be(ModelSelectionReason.Fits);
    }

    [Fact]
    public void QuickStartTakesTheSmallestWeightsAmongTheModelsThatFitTheVram()
    {
        KeyValuePair<string, GgufModelInfo>[] pool = [Entry("small", 2 * GiB, 16 * 1024), Entry("mid", 6 * GiB, 16 * 1024), Entry("big", 12 * GiB, 16 * 1024)];

        Select(pool, vramBytes: 30 * GiB, goal: AutoSelectionGoal.Quality).Selected.AliasName.Should().Be("big");
        Select(pool, vramBytes: 30 * GiB, goal: AutoSelectionGoal.QuickStart).Selected.AliasName.Should().Be("small");
    }

    [Fact]
    public void WhenNothingFitsAnywhereTheSmallestOfThePoolIsTheFallback()
    {
        KeyValuePair<string, GgufModelInfo>[] pool = [Entry("mid", 6 * GiB, 16 * 1024), Entry("small", 2 * GiB, 16 * 1024), Entry("big", 12 * GiB, 16 * 1024)];

        var result = Select(pool, vramBytes: 1 * GiB, systemRamBytes: 0);

        result.Selected.AliasName.Should().Be("small");
        result.Reason.Should().Be(ModelSelectionReason.FallbackToSmallest);
    }

    [Fact]
    public void AnEmptyPoolAnswersWithTheFallbackEntry()
    {
        var result = Select([], vramBytes: 24 * GiB);

        result.Selected.RepoId.Should().Be("example/fallback");
        result.Reason.Should().Be(ModelSelectionReason.FallbackToSmallest);
        result.Candidates.Should().BeEmpty();
    }

    [Fact]
    public void AModelWithAnExcludedKnownIssueIsNotConsidered()
    {
        KeyValuePair<string, GgufModelInfo>[] pool =
            [Entry("big-flawed", 12 * GiB, 16 * 1024, GgufModelKnownIssues.ToolUseUnreliableQ4), Entry("mid", 6 * GiB, 16 * 1024)];

        var excluded = Select(pool, vramBytes: 30 * GiB, excludeKnownIssues: [GgufModelKnownIssues.ToolUseUnreliableQ4]);
        var notExcluded = Select(pool, vramBytes: 30 * GiB);

        excluded.Selected.AliasName.Should().Be("mid");
        excluded.Candidates.Select(c => c.Model.AliasName).Should().Equal("mid");
        notExcluded.Selected.AliasName.Should().Be("big-flawed");
    }

    [Fact]
    public void AMixtureOfExpertsModelNeedsItsExtraReserveToFit()
    {
        // The reserve is the loader's own rule (GgufMemoryModel); the selection must apply it, not a copy of it. A card that holds a
        // dense model by less than the reserve does not hold the same size of mixture.
        var dense = Entry("dense", 6 * GiB, 16 * 1024).Value;
        var moe = dense with { IsMixtureOfExperts = true };
        var need = GgufMemoryModel.ResidentBytes(6 * GiB, mixtureOfExperts: false) + DefaultContext * 16 * 1024L;
        var vram = need + GgufMemoryModel.MixtureOfExpertsReserveBytes / 2;

        Select([new("dense", dense)], vramBytes: vram).Candidates.Single().Fits.Should().BeTrue();
        Select([new("moe", moe)], vramBytes: vram).Candidates.Single().Fits.Should().BeFalse();
    }
}
