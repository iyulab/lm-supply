using AwesomeAssertions;
using LMSupply.Exceptions;

namespace LMSupply.Embedder.Tests;

public class LocalEmbedderApiTests
{
    [Fact]
    public void GetAvailableModels_ReturnsKnownModels()
    {
        var models = LocalEmbedder.GetAvailableModels().ToList();

        models.Should().NotBeEmpty();
        models.Should().Contain("default");
        models.Should().Contain("fast");
        models.Should().Contain("auto");
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

    /// <summary>
    /// Regression for ISSUE-lm-supply-1775535000-multilingual-embedder-vocab:
    /// when a local model.onnx exists but no tokenizer files are present, the error must
    /// list every probed path and mention the supported tokenizer formats — not just
    /// "Vocabulary file not found".
    /// </summary>
    [Fact]
    public async Task LoadAsync_LocalModelWithoutTokenizer_ListsAllProbePaths()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"embedder-tk-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        try
        {
            var modelPath = Path.Combine(tempDir, "model.onnx");
            await File.WriteAllBytesAsync(modelPath, [0x00], TestContext.Current.CancellationToken); // placeholder so File.Exists is true

            var act = () => LocalEmbedder.LoadAsync(modelPath);

            var ex = await act.Should().ThrowAsync<ModelNotFoundException>();
            ex.Which.Message.Should().Contain("vocab.txt");
            ex.Which.Message.Should().Contain("tokenizer.json");
            ex.Which.Message.Should().Contain("sentencepiece.bpe.model");
            ex.Which.Message.Should().Contain("model"); // model id appears
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
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
    [InlineData("default")]
    [InlineData("fast")]
    [InlineData("quality")]
    [InlineData("large")]
    public void GetAvailableModels_ContainsExpectedAliases(string expectedAlias)
    {
        var models = LocalEmbedder.GetAvailableModels().ToList();
        models.Should().Contain(expectedAlias);
    }

    [Fact]
    public void CheckRuntimeAvailability_ReturnsWithoutCrash()
    {
        var (available, errorMessage) = LocalEmbedder.CheckRuntimeAvailability();

        // On dev machines this should be true; on bare containers it may be false.
        // The key invariant: the call itself must not throw or crash the process.
        if (available)
            errorMessage.Should().BeNull();
        else
            errorMessage.Should().NotBeNullOrEmpty();
    }

    [Fact]
    public void IsModelDownloaded_ReturnsFalseForNonCachedModel()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"embedder-cache-{Guid.NewGuid():N}");
        try
        {
            // Empty cache dir → no model can be downloaded
            var result = LocalEmbedder.IsModelDownloaded("default", tempDir);
            result.Should().BeFalse();
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void IsModelDownloaded_ReturnsFalseForUnknownModel()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"embedder-cache-{Guid.NewGuid():N}");
        try
        {
            var result = LocalEmbedder.IsModelDownloaded("nonexistent/model-xyz", tempDir);
            result.Should().BeFalse();
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public async Task DownloadModelAsync_ThrowsForUnknownModel()
    {
        var act = () => LocalEmbedder.DownloadModelAsync("completely-unknown-model-xyz");
        await act.Should().ThrowAsync<ModelNotFoundException>();
    }
}
