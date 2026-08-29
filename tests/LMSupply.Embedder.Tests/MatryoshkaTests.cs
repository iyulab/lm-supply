using AwesomeAssertions;
using LMSupply.Embedder.Utils;

namespace LMSupply.Embedder.Tests;

public class MatryoshkaTests : IAsyncDisposable
{
    private readonly FakeEmbeddingModel _model = new(fullDimensions: 768);

    public async ValueTask DisposeAsync()
    {
        await _model.DisposeAsync();
        GC.SuppressFinalize(this);
    }

    [Fact]
    public async Task EmbedAsync_WithDimensions_TruncatesToRequestedSize()
    {
        var result = await _model.EmbedAsync("hello", 256, TestContext.Current.CancellationToken);
        result.Should().HaveCount(256);
    }

    [Fact]
    public async Task EmbedAsync_WithDimensions_ResultIsL2Normalized()
    {
        var result = await _model.EmbedAsync("hello", 256, TestContext.Current.CancellationToken);
        var norm = MathF.Sqrt(result.Sum(v => v * v));
        norm.Should().BeApproximately(1f, 0.001f);
    }

    [Fact]
    public async Task EmbedAsync_WithFullDimensions_ReturnsSameAsBaseOverload()
    {
        var full = await _model.EmbedAsync("hello", TestContext.Current.CancellationToken);
        var withDim = await _model.EmbedAsync("hello", 768, TestContext.Current.CancellationToken);
        withDim.Should().BeEquivalentTo(full);
    }

    [Fact]
    public async Task EmbedAsync_WithDimensionsExceedingModel_ThrowsArgumentOutOfRange()
    {
        await _model.Invoking(m => m.EmbedAsync("hello", 1024).AsTask())
            .Should().ThrowAsync<ArgumentOutOfRangeException>();
    }

    [Fact]
    public async Task EmbedAsync_Batch_WithDimensions_TruncatesAll()
    {
        var results = await _model.EmbedAsync(["a", "b", "c"], 128, TestContext.Current.CancellationToken);
        results.Should().AllSatisfy(r => r.Should().HaveCount(128));
    }

    [Fact]
    public async Task EmbedAsync_Batch_WithDimensions_AllL2Normalized()
    {
        var results = await _model.EmbedAsync(["a", "b"], 128, TestContext.Current.CancellationToken);
        foreach (var r in results)
        {
            var norm = MathF.Sqrt(r.Sum(v => v * v));
            norm.Should().BeApproximately(1f, 0.001f);
        }
    }
}

internal sealed class FakeEmbeddingModel : IEmbeddingModel
{
    private readonly int _dims;

    public FakeEmbeddingModel(int fullDimensions) => _dims = fullDimensions;

    public string ModelId => "fake";
    public int Dimensions => _dims;
    public bool IsGpuActive => false;
    public IReadOnlyList<string> ActiveProviders => [];
    public ExecutionProvider RequestedProvider => ExecutionProvider.Cpu;
    public long? EstimatedMemoryBytes => null;

    public ValueTask<float[]> EmbedAsync(string text, CancellationToken ct = default)
    {
        var rng = new Random(text.GetHashCode());
        var v = Enumerable.Range(0, _dims).Select(_ => (float)rng.NextDouble()).ToArray();
        NormalizeL2(v);
        return ValueTask.FromResult(v);
    }

    public ValueTask<float[][]> EmbedAsync(IReadOnlyList<string> texts, CancellationToken ct = default)
    {
        var result = texts.Select(t =>
        {
            var rng = new Random(t.GetHashCode());
            var v = Enumerable.Range(0, _dims).Select(_ => (float)rng.NextDouble()).ToArray();
            NormalizeL2(v);
            return v;
        }).ToArray();
        return ValueTask.FromResult(result);
    }

    public async ValueTask<float[]> EmbedAsync(string text, int dimensions, CancellationToken ct = default)
    {
        if (dimensions <= 0 || dimensions > _dims)
            throw new ArgumentOutOfRangeException(nameof(dimensions));
        var full = await EmbedAsync(text, ct);
        if (dimensions == _dims) return full;
        var t = full[..dimensions];
        NormalizeL2(t);
        return t;
    }

    public async ValueTask<float[][]> EmbedAsync(IReadOnlyList<string> texts, int dimensions, CancellationToken ct = default)
    {
        if (dimensions <= 0 || dimensions > _dims)
            throw new ArgumentOutOfRangeException(nameof(dimensions));
        var full = await EmbedAsync(texts, ct);
        if (dimensions == _dims) return full;
        var result = new float[full.Length][];
        for (int i = 0; i < full.Length; i++) { result[i] = full[i][..dimensions]; NormalizeL2(result[i]); }
        return result;
    }

    public Task WarmupAsync(CancellationToken ct = default) => Task.CompletedTask;
    public ModelInfo? GetModelInfo() => null;
    public ValueTask DisposeAsync() => ValueTask.CompletedTask;

    private static void NormalizeL2(float[] v)
    {
        float norm = MathF.Sqrt(v.Sum(x => x * x));
        if (norm > 0f) for (int i = 0; i < v.Length; i++) v[i] /= norm;
    }
}
