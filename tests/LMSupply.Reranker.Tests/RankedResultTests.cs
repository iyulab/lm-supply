using FluentAssertions;

namespace LMSupply.Reranker.Tests;

public class RankedResultTests
{
    [Fact]
    public void CompareTo_HigherScore_ShouldComeFirst()
    {
        var high = new RankedResult(0, 0.9f, "high");
        var low = new RankedResult(1, 0.1f, "low");

        high.CompareTo(low).Should().BeLessThan(0);
        low.CompareTo(high).Should().BeGreaterThan(0);
    }

    [Fact]
    public void CompareTo_EqualScores_ShouldBeZero()
    {
        var a = new RankedResult(0, 0.5f, "a");
        var b = new RankedResult(1, 0.5f, "b");

        a.CompareTo(b).Should().Be(0);
    }

    [Fact]
    public void Sort_ShouldOrderByScoreDescending()
    {
        var results = new RankedResult[]
        {
            new(0, 0.3f, "low"),
            new(1, 0.9f, "high"),
            new(2, 0.5f, "mid")
        };

        Array.Sort(results);

        results[0].Score.Should().Be(0.9f);
        results[1].Score.Should().Be(0.5f);
        results[2].Score.Should().Be(0.3f);
    }

    [Fact]
    public void ToString_ShouldIncludeScoreAndPreview()
    {
        var result = new RankedResult(5, 0.8765f, "This is a test document");

        var str = result.ToString();

        str.Should().Contain("0.8765");
        str.Should().Contain("#5");
        str.Should().Contain("This is a test document");
    }

    [Fact]
    public void ToString_LongDocument_ShouldTruncate()
    {
        var longDoc = new string('x', 100);
        var result = new RankedResult(0, 0.5f, longDoc);

        var str = result.ToString();

        str.Should().Contain("...");
        str.Length.Should().BeLessThan(longDoc.Length + 50);
    }

    [Fact]
    public void RecordEquality_ShouldWork()
    {
        var a = new RankedResult(0, 0.5f, "doc");
        var b = new RankedResult(0, 0.5f, "doc");
        var c = new RankedResult(1, 0.5f, "doc");

        a.Should().Be(b);
        a.Should().NotBe(c);
    }
}
