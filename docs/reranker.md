# LMSupply.Reranker

A lightweight, zero-configuration semantic reranking library for .NET using cross-encoder models.

## Installation

```bash
dotnet add package LMSupply.Reranker
```

For GPU acceleration:

```bash
# NVIDIA CUDA
dotnet add package Microsoft.ML.OnnxRuntime.Gpu

# Windows DirectML
dotnet add package Microsoft.ML.OnnxRuntime.DirectML

# macOS CoreML
dotnet add package Microsoft.ML.OnnxRuntime.CoreML
```

## What is Reranking?

Reranking improves search quality by using a cross-encoder model to score the relevance of documents to a query. Unlike embedding-based retrieval (bi-encoder), cross-encoders see both the query and document together, producing more accurate relevance scores.

**Typical RAG Pipeline:**
1. **Retrieve** - Fast retrieval using embeddings (bi-encoder) → top 100 candidates
2. **Rerank** - Score candidates with cross-encoder → top 10 most relevant
3. **Generate** - Pass top results to LLM

## Basic Usage

```csharp
using LMSupply.Reranker;

// Load the default model
await using var reranker = await LocalReranker.LoadAsync("default");

// Rerank documents
var results = await reranker.RerankAsync(
    query: "What is machine learning?",
    documents: new[]
    {
        "Machine learning is a branch of AI that enables computers to learn from data.",
        "The weather forecast predicts rain tomorrow.",
        "Deep learning uses neural networks with many layers.",
        "Python is widely used for data science."
    },
    topK: 2  // Return top 2 results
);

foreach (var result in results)
{
    Console.WriteLine($"[{result.Score:F4}] #{result.OriginalIndex}: {result.Document}");
}
```

Output:
```
[0.9823] #0: Machine learning is a branch of AI that enables computers to learn from data.
[0.8456] #2: Deep learning uses neural networks with many layers.
```

## Available Models

| Alias | HuggingFace ID | Max Tokens | Size | Description |
|-------|---------------|------------|------|-------------|
| `default` | cross-encoder/ms-marco-MiniLM-L-6-v2 | 512 | ~90MB | Fast, good quality |
| `quality` | cross-encoder/ms-marco-MiniLM-L-12-v2 | 512 | ~134MB | Higher accuracy |
| `fast` | cross-encoder/ms-marco-TinyBERT-L-2-v2 | 512 | ~18MB | Ultra-fast |
| `multilingual` | BAAI/bge-reranker-v2-m3 | 8192 | ~1.1GB | 100+ languages, long context |
| `bge-base` | BAAI/bge-reranker-base | 512 | ~440MB | Good multilingual |

## Configuration Options

```csharp
var options = new RerankerOptions
{
    // Model to use
    ModelId = "default",

    // GPU/CPU execution provider
    Provider = ExecutionProvider.Auto,

    // Maximum input sequence length
    MaxSequenceLength = 512,

    // Batch size for processing multiple documents
    BatchSize = 32,

    // Number of inference threads (null = auto)
    ThreadCount = null,

    // Disable automatic model download
    DisableAutoDownload = false,

    // Custom cache directory
    CacheDirectory = null
};

var reranker = await LocalReranker.LoadAsync("default", options);
```

## Scoring Without Sorting

Get raw relevance scores without sorting:

```csharp
float[] scores = await reranker.ScoreAsync(
    query: "machine learning",
    documents: documents
);

// scores[i] corresponds to documents[i]
for (int i = 0; i < scores.Length; i++)
{
    Console.WriteLine($"Document {i}: {scores[i]:F4}");
}
```

## Batch Reranking

Rerank multiple query-document sets efficiently:

```csharp
var queries = new[] { "query1", "query2" };
var documentSets = new[]
{
    new[] { "doc1a", "doc1b", "doc1c" },
    new[] { "doc2a", "doc2b" }
};

var batchResults = await reranker.RerankBatchAsync(
    queries: queries,
    documentSets: documentSets,
    topK: 2
);

// batchResults[0] = results for query1
// batchResults[1] = results for query2
```

## RAG Integration Example

```csharp
using LMSupply.Embedder;
using LMSupply.Reranker;

public class RagPipeline
{
    private readonly IEmbeddingModel _embedder;
    private readonly IReranker _reranker;
    private readonly float[][] _documentEmbeddings;
    private readonly string[] _documents;

    public async Task<string[]> RetrieveAsync(string query, int topK = 5)
    {
        // Stage 1: Fast retrieval with embeddings
        var queryEmbedding = await _embedder.EmbedAsync(query);

        var candidates = _documents
            .Select((doc, i) => new
            {
                Document = doc,
                Index = i,
                Score = _embedder.CosineSimilarity(queryEmbedding, _documentEmbeddings[i])
            })
            .OrderByDescending(x => x.Score)
            .Take(20)  // Get top 20 candidates
            .ToArray();

        // Stage 2: Rerank with cross-encoder
        var reranked = await _reranker.RerankAsync(
            query: query,
            documents: candidates.Select(c => c.Document),
            topK: topK
        );

        return reranked.Select(r => r.Document).ToArray();
    }
}
```

## Score Interpretation

Scores are normalized between 0.0 and 1.0:

| Score Range | Interpretation |
|-------------|----------------|
| 0.9 - 1.0 | Highly relevant |
| 0.7 - 0.9 | Relevant |
| 0.5 - 0.7 | Somewhat relevant |
| 0.3 - 0.5 | Marginally relevant |
| 0.0 - 0.3 | Not relevant |

## Thread Safety

The `IReranker` instance is thread-safe:

```csharp
await using var reranker = await LocalReranker.LoadAsync("default");

// Safe to use concurrently
await Task.WhenAll(
    Task.Run(() => reranker.RerankAsync("query1", docs1)),
    Task.Run(() => reranker.RerankAsync("query2", docs2))
);
```

## Warmup

Avoid cold-start latency:

```csharp
await using var reranker = await LocalReranker.LoadAsync("default");
await reranker.WarmupAsync();  // Pre-loads the model
```

## Model Information

```csharp
var info = reranker.GetModelInfo();
if (info != null)
{
    Console.WriteLine($"Model: {info.DisplayName}");
    Console.WriteLine($"Max Tokens: {info.MaxSequenceLength}");
    Console.WriteLine($"Size: {info.SizeMB:F0}MB");
    Console.WriteLine($"Multilingual: {info.IsMultilingual}");
}
```

## Performance Tips

1. **Limit candidate count** - Reranking is slower than embedding retrieval; use it on top ~20-100 candidates
2. **Adjust batch size** - Larger batches are faster but use more memory
3. **Use GPU acceleration** - Install ONNX Runtime GPU package for significant speedup
4. **Choose the right model**:
   - `fast` for latency-critical applications
   - `default` for balanced performance
   - `quality` when accuracy is paramount
   - `multilingual` for non-English or mixed-language content
5. **Warmup before production** - Call `WarmupAsync()` to pre-load the model

## Handling Long Documents

For documents longer than the model's max sequence length:

```csharp
// The model automatically truncates to MaxSequenceLength
// For long documents, consider chunking:

string[] ChunkDocument(string document, int chunkSize = 500)
{
    var words = document.Split(' ');
    return words
        .Select((word, index) => new { word, index })
        .GroupBy(x => x.index / chunkSize)
        .Select(g => string.Join(' ', g.Select(x => x.word)))
        .ToArray();
}

// Rerank chunks and take the best score
var chunks = ChunkDocument(longDocument);
var scores = await reranker.ScoreAsync(query, chunks);
float bestScore = scores.Max();
```
