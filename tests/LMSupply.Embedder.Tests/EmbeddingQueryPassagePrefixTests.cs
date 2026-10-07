using AwesomeAssertions;
using LMSupply.Embedder.Utils;

namespace LMSupply.Embedder.Tests;

/// <summary>
/// Tests for IEmbeddingModel.EmbedAsync/EmbedQueryAsync/EmbedPassageAsync —
/// default interface methods that apply ModelInfo.DefaultPrefix/QueryPrefix/PassagePrefix automatically
/// over the implementor's EmbedRawAsync.
/// Uses a minimal fake implementor to isolate the default-method prefix logic from any real
/// tokenizer/model/GPU dependency. Default interface method bodies are only reachable through
/// the interface type (not the concrete class), so tests call through an `IEmbeddingModel`-typed
/// reference and assert on the concrete `RecordingEmbeddingModel` it wraps.
/// </summary>
public class EmbeddingQueryPassagePrefixTests
{
    [Fact]
    public async Task EmbedQueryAsync_AppliesQueryPrefix_WhenModelHasOne()
    {
        var recorder = new RecordingEmbeddingModel(new ModelInfo
        {
            RepoId = "intfloat/multilingual-e5-base",
            Dimensions = 768,
            MaxSequenceLength = 512,
            PoolingMode = PoolingMode.Mean,
            DoLowerCase = false,
            QueryPrefix = "query: ",
            PassagePrefix = "passage: "
        });
        IEmbeddingModel model = recorder;

        await model.EmbedQueryAsync("what is the capital of France?", TestContext.Current.CancellationToken);

        recorder.LastEmbeddedText.Should().Be("query: what is the capital of France?");
    }

    [Fact]
    public async Task EmbedPassageAsync_AppliesPassagePrefix_WhenModelHasOne()
    {
        var recorder = new RecordingEmbeddingModel(new ModelInfo
        {
            RepoId = "intfloat/multilingual-e5-base",
            Dimensions = 768,
            MaxSequenceLength = 512,
            PoolingMode = PoolingMode.Mean,
            DoLowerCase = false,
            QueryPrefix = "query: ",
            PassagePrefix = "passage: "
        });
        IEmbeddingModel model = recorder;

        await model.EmbedPassageAsync("Paris is the capital of France.", TestContext.Current.CancellationToken);

        recorder.LastEmbeddedText.Should().Be("passage: Paris is the capital of France.");
    }

    [Fact]
    public async Task EmbedQueryAsync_IsNoOpPassthrough_WhenModelHasNoPrefix()
    {
        var recorder = new RecordingEmbeddingModel(new ModelInfo
        {
            RepoId = "BAAI/bge-m3",
            Dimensions = 1024,
            MaxSequenceLength = 8192,
            PoolingMode = PoolingMode.Cls,
            DoLowerCase = false,
            QueryPrefix = null,
            PassagePrefix = null
        });
        IEmbeddingModel model = recorder;

        await model.EmbedQueryAsync("what is the capital of France?", TestContext.Current.CancellationToken);

        recorder.LastEmbeddedText.Should().Be("what is the capital of France?");
    }

    [Fact]
    public async Task EmbedQueryAsync_IsNoOpPassthrough_WhenModelInfoIsNull()
    {
        var recorder = new RecordingEmbeddingModel(modelInfo: null);
        IEmbeddingModel model = recorder;

        await model.EmbedQueryAsync("what is the capital of France?", TestContext.Current.CancellationToken);

        recorder.LastEmbeddedText.Should().Be("what is the capital of France?");
    }

    [Fact]
    public async Task EmbedQueryAsync_Batch_AppliesPrefixToEachText()
    {
        var recorder = new RecordingEmbeddingModel(new ModelInfo
        {
            RepoId = "intfloat/multilingual-e5-base",
            Dimensions = 768,
            MaxSequenceLength = 512,
            PoolingMode = PoolingMode.Mean,
            DoLowerCase = false,
            QueryPrefix = "query: ",
            PassagePrefix = "passage: "
        });
        IEmbeddingModel model = recorder;

        await model.EmbedQueryAsync(["first question", "second question"], TestContext.Current.CancellationToken);

        recorder.LastEmbeddedBatch.Should().Equal("query: first question", "query: second question");
    }

    [Fact]
    public async Task EmbedPassageAsync_WithDimensions_AppliesPrefixAndForwardsDimensions()
    {
        var recorder = new RecordingEmbeddingModel(new ModelInfo
        {
            RepoId = "nomic-ai/nomic-embed-text-v1.5",
            Dimensions = 768,
            MaxSequenceLength = 8192,
            PoolingMode = PoolingMode.Mean,
            DoLowerCase = false,
            QueryPrefix = "search_query: ",
            PassagePrefix = "search_document: "
        });
        IEmbeddingModel model = recorder;

        await model.EmbedPassageAsync("Matryoshka embeddings support truncation.", 256, TestContext.Current.CancellationToken);

        recorder.LastEmbeddedText.Should().Be("search_document: Matryoshka embeddings support truncation.");
        recorder.LastDimensions.Should().Be(256);
    }

    private static ModelInfo E5Info() => new()
    {
        RepoId = "intfloat/multilingual-e5-base",
        Dimensions = 768,
        MaxSequenceLength = 512,
        PoolingMode = PoolingMode.Mean,
        DoLowerCase = false,
        QueryPrefix = "query: ",
        PassagePrefix = "passage: ",
        DefaultPrefix = "query: "
    };

    [Fact]
    public async Task EmbedAsync_AppliesDefaultPrefix_WhenModelHasOne()
    {
        var recorder = new RecordingEmbeddingModel(E5Info());
        IEmbeddingModel model = recorder;

        await model.EmbedAsync("two sentences to compare", TestContext.Current.CancellationToken);

        recorder.LastEmbeddedText.Should().Be("query: two sentences to compare");
    }

    [Fact]
    public async Task EmbedAsync_Batch_WithDimensions_AppliesDefaultPrefixAndForwardsDimensions()
    {
        var recorder = new RecordingEmbeddingModel(E5Info());
        IEmbeddingModel model = recorder;

        await model.EmbedAsync(["a", "b"], 128, TestContext.Current.CancellationToken);

        recorder.LastEmbeddedBatch.Should().Equal("query: a", "query: b");
        recorder.LastDimensions.Should().Be(128);
    }

    [Fact]
    public async Task EmbedAsync_IsRaw_WhenModelHasNoDefaultPrefix()
    {
        var recorder = new RecordingEmbeddingModel(E5Info() with { DefaultPrefix = null });
        IEmbeddingModel model = recorder;

        await model.EmbedAsync("as given", TestContext.Current.CancellationToken);

        recorder.LastEmbeddedText.Should().Be("as given");
    }

    [Fact]
    public async Task EmbedRawAsync_AppliesNoPrefix_EvenWhenModelDeclaresAll()
    {
        var recorder = new RecordingEmbeddingModel(E5Info());
        IEmbeddingModel model = recorder;

        await model.EmbedRawAsync("query: my own instruction", TestContext.Current.CancellationToken);

        recorder.LastEmbeddedText.Should().Be("query: my own instruction");
    }

    /// <summary>
    /// Minimal IEmbeddingModel fake that records what text/dimensions the default interface
    /// methods (EmbedAsync/EmbedQueryAsync/EmbedPassageAsync) forwarded to EmbedRawAsync, instead of running
    /// real inference.
    /// </summary>
    private sealed class RecordingEmbeddingModel(ModelInfo? modelInfo) : IEmbeddingModel
    {
        public string? LastEmbeddedText { get; private set; }
        public IReadOnlyList<string>? LastEmbeddedBatch { get; private set; }
        public int? LastDimensions { get; private set; }

        public string ModelId => "recording-fake";
        public int Dimensions => modelInfo?.Dimensions ?? 0;
        public bool IsGpuActive => false;
        public IReadOnlyList<string> ActiveProviders => [];
        public ExecutionProvider RequestedProvider => ExecutionProvider.Cpu;
        public long? EstimatedMemoryBytes => null;

        public ValueTask<float[]> EmbedRawAsync(string text, CancellationToken cancellationToken = default)
        {
            LastEmbeddedText = text;
            return ValueTask.FromResult(new float[Dimensions]);
        }

        public ValueTask<float[][]> EmbedRawAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default)
        {
            LastEmbeddedBatch = texts;
            return ValueTask.FromResult(texts.Select(_ => new float[Dimensions]).ToArray());
        }

        public ValueTask<float[]> EmbedRawAsync(string text, int dimensions, CancellationToken cancellationToken = default)
        {
            LastEmbeddedText = text;
            LastDimensions = dimensions;
            return ValueTask.FromResult(new float[dimensions]);
        }

        public ValueTask<float[][]> EmbedRawAsync(IReadOnlyList<string> texts, int dimensions, CancellationToken cancellationToken = default)
        {
            LastEmbeddedBatch = texts;
            LastDimensions = dimensions;
            return ValueTask.FromResult(texts.Select(_ => new float[dimensions]).ToArray());
        }

        public Task WarmupAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
        public ModelInfo? GetModelInfo() => modelInfo;
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }
}
