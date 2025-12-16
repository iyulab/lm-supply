using FluentAssertions;
using LMSupply.Exceptions;

namespace LMSupply.Embedder.Tests;

public class LocalEmbedderApiTests
{
    [Fact]
    public void GetAvailableModels_ReturnsKnownModels()
    {
        var models = LocalEmbedder.GetAvailableModels().ToList();

        models.Should().NotBeEmpty();
        models.Should().Contain("all-MiniLM-L6-v2");
    }

    [Fact]
    public void CosineSimilarity_ComputesCorrectly()
    {
        var vec1 = new float[] { 1, 0, 0 };
        var vec2 = new float[] { 1, 0, 0 };

        var result = LocalEmbedder.CosineSimilarity(vec1, vec2);
        result.Should().BeApproximately(1.0f, 0.00001f);
    }

    [Fact]
    public void EuclideanDistance_ComputesCorrectly()
    {
        var vec1 = new float[] { 0, 0 };
        var vec2 = new float[] { 3, 4 };

        var result = LocalEmbedder.EuclideanDistance(vec1, vec2);
        result.Should().BeApproximately(5.0f, 0.00001f);
    }

    [Fact]
    public void DotProduct_ComputesCorrectly()
    {
        var vec1 = new float[] { 1, 2, 3 };
        var vec2 = new float[] { 4, 5, 6 };

        var result = LocalEmbedder.DotProduct(vec1, vec2);
        result.Should().BeApproximately(32.0f, 0.00001f);
    }

    [Fact]
    public async Task LoadAsync_ThrowsForUnknownModel()
    {
        var act = () => LocalEmbedder.LoadAsync("completely-unknown-model-xyz");
        await act.Should().ThrowAsync<ModelNotFoundException>();
    }

    [Fact]
    public async Task LoadAsync_ThrowsForMissingLocalFile()
    {
        var act = () => LocalEmbedder.LoadAsync("/nonexistent/path/model.onnx");
        await act.Should().ThrowAsync<ModelNotFoundException>();
    }

    [Fact]
    public async Task LoadAsync_AcceptsCustomOptions()
    {
        var options = new EmbedderOptions
        {
            MaxSequenceLength = 128,
            Provider = ExecutionProvider.Cpu
        };

        // This will fail because model doesn't exist, but options should be accepted
        var act = () => LocalEmbedder.LoadAsync("unknown", options);
        await act.Should().ThrowAsync<ModelNotFoundException>();
    }

    [Fact]
    public void CosineSimilarity_ThrowsOnEmptyVectors()
    {
        var empty = Array.Empty<float>();
        var vec = new float[] { 1, 2, 3 };

        var act = () => LocalEmbedder.CosineSimilarity(empty, vec);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void CosineSimilarity_ThrowsOnMismatchedLengths()
    {
        var vec1 = new float[] { 1, 2, 3 };
        var vec2 = new float[] { 1, 2 };

        var act = () => LocalEmbedder.CosineSimilarity(vec1, vec2);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void EuclideanDistance_ThrowsOnMismatchedLengths()
    {
        var vec1 = new float[] { 1, 2, 3 };
        var vec2 = new float[] { 1, 2, 3, 4 };

        var act = () => LocalEmbedder.EuclideanDistance(vec1, vec2);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void DotProduct_ThrowsOnEmptyVectors()
    {
        var empty = Array.Empty<float>();
        var vec = new float[] { 1, 2, 3 };

        var act = () => LocalEmbedder.DotProduct(empty, vec);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void CosineSimilarity_WorksWithLargeVectors()
    {
        var vec1 = Enumerable.Range(0, 384).Select(i => (float)i).ToArray();
        var vec2 = Enumerable.Range(0, 384).Select(i => (float)i).ToArray();

        var result = LocalEmbedder.CosineSimilarity(vec1, vec2);
        result.Should().BeApproximately(1.0f, 0.00001f);
    }

    [Theory]
    [InlineData("all-MiniLM-L6-v2")]
    [InlineData("all-mpnet-base-v2")]
    [InlineData("bge-small-en-v1.5")]
    public void GetAvailableModels_ContainsExpectedModels(string expectedModel)
    {
        var models = LocalEmbedder.GetAvailableModels().ToList();
        models.Should().Contain(expectedModel);
    }
}
