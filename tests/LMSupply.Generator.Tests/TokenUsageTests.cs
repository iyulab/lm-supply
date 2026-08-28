using AwesomeAssertions;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.Tests;

public class TokenUsageTests
{
    [Fact]
    public void EstimateTokens_EmptyString_ReturnsZero()
    {
        TokenUsage.EstimateTokens("").Should().Be(0);
    }

    [Fact]
    public void EstimateTokens_NullString_ReturnsZero()
    {
        TokenUsage.EstimateTokens(null!).Should().Be(0);
    }

    [Fact]
    public void EstimateTokens_ShortText_ReturnsPositive()
    {
        var count = TokenUsage.EstimateTokens("hello world");
        count.Should().BeGreaterThan(0);
    }

    [Fact]
    public void EstimateTokens_LongerText_ReturnsMoreTokens()
    {
        var shortCount = TokenUsage.EstimateTokens("hello");
        var longCount = TokenUsage.EstimateTokens("hello world this is a longer sentence with more tokens");

        longCount.Should().BeGreaterThan(shortCount);
    }

    [Fact]
    public void EstimateTokens_ApproximatelyFourCharsPerToken()
    {
        // 20 chars → ~5 tokens (4 chars/token heuristic)
        var count = TokenUsage.EstimateTokens("12345678901234567890");
        count.Should().Be(5);
    }

    [Fact]
    public void TokenUsage_TotalTokens_IsSumOfPromptAndCompletion()
    {
        var usage = new TokenUsage(10, 20);
        usage.TotalTokens.Should().Be(30);
    }

    [Fact]
    public void TokenUsage_Empty_HasZeroCounts()
    {
        var usage = TokenUsage.Empty;
        usage.PromptTokens.Should().Be(0);
        usage.CompletionTokens.Should().Be(0);
        usage.TotalTokens.Should().Be(0);
    }
}
