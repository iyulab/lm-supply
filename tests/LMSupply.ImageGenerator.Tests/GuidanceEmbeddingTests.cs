using AwesomeAssertions;
using LMSupply.ImageGenerator.Pipeline;

namespace LMSupply.ImageGenerator.Tests;

public sealed class GuidanceEmbeddingTests
{
    [Fact]
    public void Create_WithoutGuidance_IsZeroSinesThenOneCosines()
    {
        var embedding = GuidanceEmbedding.Create(1.0f, 256);

        embedding.Should().HaveCount(256);
        embedding.Take(128).Should().OnlyContain(value => value == 0.0f);
        embedding.Skip(128).Should().OnlyContain(value => value == 1.0f);
    }

    [Theory]
    [InlineData(0, -0.8512359f)]
    [InlineData(1, 0.8421312f)]
    [InlineData(2, 0.011876258f)]
    [InlineData(64, -0.072437547f)]
    [InlineData(127, 0.68163872f)]
    [InlineData(128, -0.52478325f)]
    [InlineData(129, 0.53927273f)]
    [InlineData(255, 0.73168892f)]
    public void Create_WithTheReferenceDefaultGuidance_GivesTheReferenceValues(int index, float expected)
    {
        // The reference embeds w = guidance - 1 = 7.5, scaled by 1000, over 256 values.
        GuidanceEmbedding.Create(8.5f, 256)[index].Should().BeApproximately(expected, 5e-3f);
    }

    [Fact]
    public void Create_WithASmallDimension_GivesSinesThenCosinesOfTheFrequencies()
    {
        float[] expected = [0.9300395f, -0.9880430f, -0.9196847f, 0.1986693f, -0.3674595f, 0.1541787f, -0.3926577f, 0.9800666f];

        var embedding = GuidanceEmbedding.Create(3.0f, 8);

        for (var i = 0; i < expected.Length; i++)
        {
            embedding[i].Should().BeApproximately(expected[i], 5e-3f);
        }
    }

    [Fact]
    public void Create_WithATooSmallDimension_Throws()
    {
        var act = () => GuidanceEmbedding.Create(8.5f, 1);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}
