using AwesomeAssertions;
using LMSupply.Embedder.Utils;
using Microsoft.Extensions.AI;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// A loaded model is usable wherever the Microsoft.Extensions.AI embedding contract is taken: vectors come back in input
/// order, the model's query/passage conventions are applied when asked for, <c>Dimensions</c> is a Matryoshka truncation,
/// and a request the model cannot serve fails instead of being answered with something else.
/// </summary>
public sealed class LocalEmbeddingGeneratorTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    [Fact]
    public async Task Vectors_ComeBackInInputOrder_WithTheModelId()
    {
        var model = new RecordingModel();
        using var generator = model.AsEmbeddingGenerator(EmbeddingTextKind.Raw);

        var result = await generator.GenerateAsync(["a", "bb", "ccc"], cancellationToken: Ct);

        result.Select(e => e.Vector.Span[0]).Should().Equal(1f, 2f, 3f);
        result.Should().OnlyContain(e => e.ModelId == "test-e5");
        model.LastTexts.Should().Equal("a", "bb", "ccc");
    }

    [Theory]
    [InlineData(EmbeddingTextKind.Default, "query: x")]
    [InlineData(EmbeddingTextKind.Query, "query: x")]
    [InlineData(EmbeddingTextKind.Passage, "passage: x")]
    [InlineData(EmbeddingTextKind.Raw, "x")]
    public async Task TextKind_AppliesTheModelsConvention(EmbeddingTextKind kind, string expected)
    {
        var model = new RecordingModel();
        using var generator = model.AsEmbeddingGenerator(kind);

        await generator.GenerateAsync(["x"], cancellationToken: Ct);

        model.LastTexts.Should().Equal(expected);
    }

    [Fact]
    public async Task Dimensions_IsServedAsAMatryoshkaTruncation()
    {
        var model = new RecordingModel();
        using var generator = model.AsEmbeddingGenerator(EmbeddingTextKind.Passage);

        var result = await generator.GenerateAsync(["x"], new EmbeddingGenerationOptions { Dimensions = 2 }, Ct);

        model.LastDimensions.Should().Be(2);
        model.LastTexts.Should().Equal("passage: x");
        result[0].Vector.Length.Should().Be(2);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(5)]
    public async Task Dimensions_OutsideTheModel_Throws(int dimensions)
    {
        using var generator = new RecordingModel().AsEmbeddingGenerator();

        var act = () => generator.GenerateAsync(["x"], new EmbeddingGenerationOptions { Dimensions = dimensions }, Ct);

        await act.Should().ThrowAsync<ArgumentOutOfRangeException>();
    }

    [Fact]
    public async Task AnotherModelId_Throws_InsteadOfEmbeddingWithThisModel()
    {
        using var generator = new RecordingModel().AsEmbeddingGenerator();

        var act = () => generator.GenerateAsync(["x"], new EmbeddingGenerationOptions { ModelId = "other" }, Ct);

        await act.Should().ThrowAsync<ArgumentException>().WithMessage("*test-e5*other*");
        (await generator.GenerateAsync(["x"], new EmbeddingGenerationOptions { ModelId = "TEST-E5" }, Ct))
            .Should().HaveCount(1, "the model's own id (any case) is accepted");
    }

    [Fact]
    public void Metadata_AndServices_DescribeTheModel()
    {
        var model = new RecordingModel();
        using var generator = model.AsEmbeddingGenerator();

        generator.Metadata.ProviderName.Should().Be("LMSupply");
        generator.Metadata.DefaultModelId.Should().Be("test-e5");
        generator.Metadata.DefaultModelDimensions.Should().Be(4);
        generator.GetService<EmbeddingGeneratorMetadata>().Should().BeSameAs(generator.Metadata);
        generator.GetService<IEmbeddingModel>().Should().BeSameAs(model);
        generator.GetService(typeof(IEmbeddingModel), serviceKey: "k").Should().BeNull();
    }

    [Fact]
    public async Task Disposing_TheGenerator_LeavesTheModelLoaded()
    {
        var model = new RecordingModel();
        var generator = model.AsEmbeddingGenerator();

        generator.Dispose();

        model.Disposed.Should().BeFalse();
        (await generator.GenerateAsync(["x"], cancellationToken: Ct)).Should().HaveCount(1);
    }

    [Fact]
    public async Task EmptyInput_ReturnsNothing_WithoutCallingTheModel()
    {
        var model = new RecordingModel();
        using var generator = model.AsEmbeddingGenerator();

        (await generator.GenerateAsync([], cancellationToken: Ct)).Should().BeEmpty();
        model.LastTexts.Should().BeNull();
    }

    /// <summary>A four-dimensional model whose i-th vector starts with the i-th text's length; E5-style prefixes.</summary>
    private sealed class RecordingModel : IEmbeddingModel
    {
        public IReadOnlyList<string>? LastTexts { get; private set; }
        public int? LastDimensions { get; private set; }
        public bool Disposed { get; private set; }

        public string ModelId => "test-e5";
        public int Dimensions => 4;
        public bool IsGpuActive => false;
        public IReadOnlyList<string> ActiveProviders => ["CPUExecutionProvider"];
        public ExecutionProvider RequestedProvider => ExecutionProvider.Cpu;
        public long? EstimatedMemoryBytes => null;

        public ModelInfo GetModelInfo() => new()
        {
            RepoId = "test/e5",
            Dimensions = 4,
            MaxSequenceLength = 512,
            PoolingMode = PoolingMode.Mean,
            DoLowerCase = false,
            QueryPrefix = "query: ",
            PassagePrefix = "passage: ",
            DefaultPrefix = "query: ",
        };

        public ValueTask<float[]> EmbedRawAsync(string text, CancellationToken cancellationToken = default) =>
            ValueTask.FromResult(Vector(text, Dimensions));

        public ValueTask<float[][]> EmbedRawAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default)
        {
            LastTexts = texts;
            return ValueTask.FromResult(texts.Select(t => Vector(t, Dimensions)).ToArray());
        }

        public ValueTask<float[]> EmbedRawAsync(string text, int dimensions, CancellationToken cancellationToken = default) =>
            ValueTask.FromResult(Vector(text, dimensions));

        public ValueTask<float[][]> EmbedRawAsync(IReadOnlyList<string> texts, int dimensions, CancellationToken cancellationToken = default)
        {
            LastTexts = texts;
            LastDimensions = dimensions;
            return ValueTask.FromResult(texts.Select(t => Vector(t, dimensions)).ToArray());
        }

        public Task WarmupAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;

        public ValueTask DisposeAsync()
        {
            Disposed = true;
            return ValueTask.CompletedTask;
        }

        private static float[] Vector(string text, int dimensions)
        {
            var v = new float[dimensions];
            v[0] = text.Length;
            return v;
        }
    }
}
