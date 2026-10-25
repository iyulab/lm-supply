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
    public void EstimateTokens_ChargesEachKindOfCharacter()
    {
        // o200k_base ratios: ~1.3 digits per token (20 → 16), ~7 Latin letters per token, ~1.35 Hangul syllables per token
        TokenUsage.EstimateTokens("12345678901234567890").Should().Be(16);
        TokenUsage.EstimateTokens("abcdefghijklmn").Should().Be(2);
        // Korean is not read as English: 10 syllables are ~8 tokens, not ceil(10 / 4) = 3
        TokenUsage.EstimateTokens("안녕하세요반갑습니다").Should().Be(8);
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
