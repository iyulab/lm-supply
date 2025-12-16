using FluentAssertions;

namespace LMSupply.Segmenter.Tests;

public class SegmentationResultTests
{
    [Fact]
    public void SegmentationResult_ShouldStoreAllProperties()
    {
        var result = new SegmentationResult
        {
            Width = 100,
            Height = 50,
            ClassMap = new int[5000],
            ConfidenceMap = new float[5000]
        };

        result.Width.Should().Be(100);
        result.Height.Should().Be(50);
        result.ClassMap.Should().HaveCount(5000);
        result.ConfidenceMap.Should().HaveCount(5000);
    }

    [Fact]
    public void GetClassAt_ValidCoordinates_ShouldReturnCorrectClass()
    {
        var classMap = new int[100];
        classMap[25] = 5; // y=0, x=25 with width=100

        var result = new SegmentationResult
        {
            Width = 100,
            Height = 1,
            ClassMap = classMap,
            ConfidenceMap = new float[100]
        };

        result.GetClassAt(25, 0).Should().Be(5);
    }

    [Fact]
    public void GetClassAt_InvalidCoordinates_ShouldThrow()
    {
        var result = new SegmentationResult
        {
            Width = 100,
            Height = 50,
            ClassMap = new int[5000],
            ConfidenceMap = new float[5000]
        };

        var act = () => result.GetClassAt(100, 0);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void GetConfidenceAt_ValidCoordinates_ShouldReturnCorrectConfidence()
    {
        var confidenceMap = new float[100];
        confidenceMap[50] = 0.95f;

        var result = new SegmentationResult
        {
            Width = 100,
            Height = 1,
            ClassMap = new int[100],
            ConfidenceMap = confidenceMap
        };

        result.GetConfidenceAt(50, 0).Should().Be(0.95f);
    }

    [Fact]
    public void UniqueClassCount_ShouldReturnCorrectCount()
    {
        var classMap = new int[] { 0, 0, 1, 1, 2, 2, 3, 3, 0, 1 };

        var result = new SegmentationResult
        {
            Width = 10,
            Height = 1,
            ClassMap = classMap,
            ConfidenceMap = new float[10]
        };

        result.UniqueClassCount.Should().Be(4);
    }

    [Fact]
    public void GetClassMask_ShouldReturnCorrectMask()
    {
        var classMap = new int[] { 0, 1, 0, 1, 2, 1, 0, 2, 1, 0 };

        var result = new SegmentationResult
        {
            Width = 10,
            Height = 1,
            ClassMap = classMap,
            ConfidenceMap = new float[10]
        };

        var mask = result.GetClassMask(1);

        mask.Should().BeEquivalentTo(new[] { false, true, false, true, false, true, false, false, true, false });
    }

    [Fact]
    public void GetClassPixelCounts_ShouldReturnCorrectCounts()
    {
        var classMap = new int[] { 0, 0, 0, 1, 1, 2 };

        var result = new SegmentationResult
        {
            Width = 6,
            Height = 1,
            ClassMap = classMap,
            ConfidenceMap = new float[6]
        };

        var counts = result.GetClassPixelCounts();

        counts[0].Should().Be(3);
        counts[1].Should().Be(2);
        counts[2].Should().Be(1);
    }

    [Fact]
    public void GetClassCoveragePercentages_ShouldReturnCorrectPercentages()
    {
        var classMap = new int[] { 0, 0, 0, 0, 1, 1 }; // 66.67% class 0, 33.33% class 1

        var result = new SegmentationResult
        {
            Width = 6,
            Height = 1,
            ClassMap = classMap,
            ConfidenceMap = new float[6]
        };

        var percentages = result.GetClassCoveragePercentages();

        percentages[0].Should().BeApproximately(66.67f, 0.01f);
        percentages[1].Should().BeApproximately(33.33f, 0.01f);
    }
}
