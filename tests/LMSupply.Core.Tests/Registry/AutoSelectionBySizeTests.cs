using AwesomeAssertions;
using LMSupply.Hardware;

namespace LMSupply.Core.Tests.Registry;

public class AutoSelectionBySizeTests
{
    private const long Mb = 1024 * 1024;

    private sealed record Candidate(string Id, long? EstimatedSizeBytes, long ParameterCount = 0) : IModelInfoBase, IModelMemoryInfo
    {
        public string AliasName => Id;

        public string? Description => null;

        public string? QuantizationType => null;
    }

    private static readonly Candidate[] LargestFirst =
    [
        new("large", 2_000 * Mb),
        new("medium", 500 * Mb),
        new("small", 100 * Mb),
    ];

    [Theory]
    [InlineData(4_000, "large")]
    [InlineData(1_000, "medium")]
    [InlineData(100, "small")]
    public void TheLargestCandidateThatFits_IsSelected(long budgetMb, string expected)
    {
        var selected = ModelRegistryBase<Candidate>.SelectLargestFitting(LargestFirst, budgetMb * Mb, out var fits);

        selected.Id.Should().Be(expected);
        fits.Should().BeTrue();
    }

    [Fact]
    public void WhenNothingFits_TheSmallestIsSelected_AndSaidNotToFit()
    {
        var selected = ModelRegistryBase<Candidate>.SelectLargestFitting(LargestFirst, 50 * Mb, out var fits);

        selected.Id.Should().Be("small");
        fits.Should().BeFalse();
    }

    [Fact]
    public void ACandidateWithoutSizeMetadata_IsACatalogDefect_NotAFit()
    {
        // An unknown size estimates to zero, which would fit every budget and win on every host
        Candidate[] candidates = [new("unsized", null), .. LargestFirst];

        var act = () => ModelRegistryBase<Candidate>.SelectLargestFitting(candidates, 50 * Mb, out _);

        act.Should().Throw<InvalidOperationException>().WithMessage("*unsized*");
    }

    [Fact]
    public void AParameterCount_IsEnoughMetadata()
    {
        Candidate[] candidates = [new("by-params", null, ParameterCount: 100_000_000)];

        ModelRegistryBase<Candidate>.SelectLargestFitting(candidates, 4_000 * Mb, out var fits).Id.Should().Be("by-params");
        fits.Should().BeTrue();
    }
}
