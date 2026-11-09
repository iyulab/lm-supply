using System.Security.Cryptography;
using System.Text;
using AwesomeAssertions;
using LMSupply.Generator.Gguf;
using LMSupply.Runtime;
using Xunit;

namespace LMSupply.Generator.Tests;

/// <summary>
/// <see cref="GgufModelRegistry.Recommend(long, long, int, IReadOnlyCollection{string}?, AutoSelectionGoal)"/> reads the catalog's ladder for
/// one machine: the order, the fit of every rung, and three markers. These tests pin what an application relies on to drop
/// its own tier table — the list and the automatic pick agree, the markers follow the judgements, and a larger machine never
/// loses a rung.
/// </summary>
public class GgufModelRecommendTests
{
    private const long GiB = 1024L * 1024 * 1024;

    private static readonly long[] VramGiB = [0, 3, 6, 8, 12, 16, 24, 48, 96, 192, 512];

    [Fact]
    public void TheLadder_IsTheTieredEntries_SmallestFirst()
    {
        var ladder = GgufModelRegistry.Recommend(512 * GiB, 0, 4096).Models;

        ladder.Select(m => m.Tier).Should().Equal(ModelTier.Fast, ModelTier.Default, ModelTier.Balanced, ModelTier.Quality, ModelTier.ExtraLarge);
        ladder.Select(m => m.Judgement.TotalBytes).Should().BeInAscendingOrder();
        ladder.Should().OnlyContain(m => GgufModelRegistry.Resolve(m.Alias) != null);
        GgufModelRegistry.GetAllModels().Count(m => m.Tier is not null).Should().Be(ladder.Count);
    }

    [Fact]
    public void TheAutomaticPick_IsAlwaysOneRungOfTheLadder_AndIsWhatAutoSelectionPicks()
    {
        foreach (var vram in VramGiB)
        {
            var gpu = new GpuInfo
            {
                Vendor = GpuVendor.Nvidia,
                DeviceName = "Test GPU",
                TotalMemoryBytes = vram * GiB,
                FreeMemoryBytes = vram * GiB,
            };
            var selection = GgufModelRegistry.GetAutoSelection(gpu, 32 * GiB, 4096, excludeKnownIssues: null);

            var recommendation = GgufModelRegistry.Recommend(selection.AvailableVramBytes, 32 * GiB, 4096);

            recommendation.Models.Where(m => m.IsAutomaticPick).Select(m => m.Alias)
                .Should().Equal([selection.Selected.AliasName], $"a {vram} GiB GPU: the list and Auto must not disagree");
        }
    }

    [Fact]
    public void TheMarkers_FollowTheJudgements()
    {
        foreach (var vram in VramGiB)
        foreach (var ram in new[] { 0L, 16 * GiB, 128 * GiB })
        {
            var models = GgufModelRegistry.Recommend(vram * GiB, ram, 4096).Models;
            var because = $"VRAM {vram} GiB, RAM {ram / GiB} GiB";

            var smallest = models.FirstOrDefault(m => m.IsUsable);
            var best = models.LastOrDefault(m => m.Judgement.Fits) ?? models.LastOrDefault(m => m.Judgement.FitsInSystemRam);

            models.Where(m => m.IsSmallestUsable).Select(m => m.Alias).Should().Equal(smallest is null ? [] : [smallest.Alias], because);
            models.Where(m => m.IsBestThatRunsWell).Select(m => m.Alias).Should().Equal(best is null ? [] : [best.Alias], because);
        }
    }

    [Fact]
    public void ALargerMachine_NeverLosesARung_AndTheBestGrowsToTheTop()
    {
        var bestIndex = new List<int>();
        foreach (var vram in VramGiB)
        {
            var models = GgufModelRegistry.Recommend(vram * GiB, 0, 4096).Models;
            bestIndex.Add(models.ToList().FindIndex(m => m.IsBestThatRunsWell));
        }

        bestIndex.Should().BeInAscendingOrder("more memory can only raise the largest rung that runs well");
        bestIndex[0].Should().Be(-1, "no GPU and no RAM budget: nothing runs");
        bestIndex[^1].Should().Be(4, "a machine with 512 GiB reaches the top tier");
    }

    [Fact]
    public void ARungThatDoesNotFit_StaysInTheListMarkedTooLarge()
    {
        var models = GgufModelRegistry.Recommend(availableVramBytes: 3 * GiB, systemRamBytes: 0, contextLength: 4096).Models;

        models.Should().HaveCount(5);
        models[^1].Judgement.Fits.Should().BeFalse();
        models[^1].Judgement.FitsInSystemRam.Should().BeFalse();
        models[^1].IsUsable.Should().BeFalse();
    }

    [Fact]
    public void ExcludingAKnownIssue_DropsTheRungsThatCarryIt_AndTheAutomaticPickStaysOnTheList()
    {
        var all = GgufModelRegistry.Recommend(24 * GiB, 32 * GiB, 4096);
        var without = GgufModelRegistry.Recommend(24 * GiB, 32 * GiB, 4096, [GgufModelKnownIssues.ThinkingEnabledByDefault]);

        without.Models.Count.Should().BeLessThan(all.Models.Count);
        without.Models.Should().OnlyContain(m => !m.Model.KnownIssues.Contains(GgufModelKnownIssues.ThinkingEnabledByDefault));
        without.Models.Count(m => m.IsAutomaticPick).Should().Be(1);
    }

    [Fact]
    public void TheNames_ArePeopleNames_WithoutTheQuantizationOrANote()
    {
        var models = GgufModelRegistry.Recommend(512 * GiB, 0, 4096).Models;

        models.Should().OnlyContain(m => !m.DisplayName.Contains('('));
        models.Single(m => m.Tier == ModelTier.Quality).DisplayName.Should().Be("Qwen 3.6 35B A3B Instruct");
        models.Single(m => m.Tier == ModelTier.Quality).Model.QuantizationType.Should().Be("IQ4_XS");
        models.Single(m => m.Tier == ModelTier.ExtraLarge).DisplayName.Should().Be("Qwen 3.5 122B A10B");
    }

    [Fact]
    public void ACpuRecommendation_JudgesSystemMemoryAlone()
    {
        var recommendation = GgufModelRegistry.Recommend(ExecutionProvider.Cpu);

        recommendation.AvailableVramBytes.Should().Be(0);
        recommendation.Models.Should().OnlyContain(m => !m.Judgement.Fits);
    }

    [Fact]
    public void TheCatalogRevision_IsTheHashOfTheRecordedSnapshot()
    {
        var recorded = File.ReadAllLines(Path.Combine(AppContext.BaseDirectory, "catalog.snapshot"))
            .Where(l => l.Length > 0 && !l.StartsWith('#'));
        var expected = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(string.Join('\n', recorded) + '\n')))[..16].ToLowerInvariant();

        GeneratorCatalog.Revision.Should().MatchRegex("^[0-9a-f]{16}$").And.Be(expected,
            "the revision changes when catalog.snapshot would, and only then");
        GgufModelRegistry.Recommend(8 * GiB, 0, 4096).CatalogRevision.Should().Be(GeneratorCatalog.Revision);
    }
}
