using AwesomeAssertions;
using LMSupply.Generator.Gguf;
using LMSupply.Runtime;
using Xunit;

namespace LMSupply.Generator.Tests;

/// <summary>
/// <see cref="GgufModelRegistry.Evaluate(GgufModelInfo, long, long, int)"/> judges any model with the rule auto-selection
/// applies to its pool, so an application can show «runs here / runs from memory / too large» for every catalog entry, not
/// only the pool's.
/// </summary>
public class GgufModelEvaluateTests
{
    private const long GiB = 1024L * 1024 * 1024;

    [Fact]
    public void APoolCandidate_IsJudgedExactlyAsAutoSelectionJudgedIt()
    {
        var gpu = new GpuInfo
        {
            Vendor = GpuVendor.Nvidia,
            DeviceName = "Test GPU",
            TotalMemoryBytes = 12 * GiB,
            FreeMemoryBytes = 11 * GiB,
        };
        var selection = GgufModelRegistry.GetAutoSelection(gpu, 32 * GiB, 8192, excludeKnownIssues: null);

        foreach (var candidate in selection.Candidates)
        {
            GgufModelRegistry.Evaluate(candidate.Model, selection.AvailableVramBytes, 32 * GiB, 8192)
                .Should().BeEquivalentTo(candidate);
        }
    }

    [Fact]
    public void AModelOutsideThePool_IsJudgedToo()
    {
        var pool = GgufModelRegistry.GetAutoSelectionAliases().Select(a => GgufModelRegistry.Resolve(a)!).ToList();
        var outside = GgufModelRegistry.GetAllModels()
            .First(m => !pool.Any(p => p.RepoId == m.RepoId && p.DefaultFile == m.DefaultFile));

        var roomy = GgufModelRegistry.Evaluate(outside, availableVramBytes: 512 * GiB, systemRamBytes: 0, contextLength: 4096);
        var ramOnly = GgufModelRegistry.Evaluate(outside, availableVramBytes: 0, systemRamBytes: 1024 * GiB, contextLength: 4096);
        var tiny = GgufModelRegistry.Evaluate(outside, availableVramBytes: 1, systemRamBytes: 0, contextLength: 4096);

        roomy.Fits.Should().BeTrue();
        ramOnly.Fits.Should().BeFalse();
        ramOnly.FitsInSystemRam.Should().BeTrue();
        tiny.Fits.Should().BeFalse();
        tiny.FitsInSystemRam.Should().BeFalse("no RAM budget was given");
        roomy.WeightsBytes.Should().BeGreaterThan(0);
    }

    [Fact]
    public void AModelDescribedByItsOwnInfo_IsJudgedFromItsSize()
    {
        var typed = new GgufModelInfo
        {
            RepoId = "example-org/typed-model-GGUF",
            DefaultFile = "typed-model.Q4_K_M.gguf",
            DisplayName = "typed",
            ChatFormat = "chatml",
            ContextLength = 8192,
            License = LicenseTier.MIT,
            EstimatedSizeBytes = 6 * GiB,
        };

        GgufModelRegistry.Evaluate(typed, availableVramBytes: 8 * GiB, systemRamBytes: 0, contextLength: 4096).Fits.Should().BeTrue();
        GgufModelRegistry.Evaluate(typed, availableVramBytes: 4 * GiB, systemRamBytes: 0, contextLength: 4096).Fits.Should().BeFalse();
    }
}
