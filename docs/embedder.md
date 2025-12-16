# LMSupply.Embedder

A lightweight, zero-configuration text embedding library for .NET with automatic GPU acceleration.

## Installation

```bash
dotnet add package LMSupply.Embedder
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

## Basic Usage

```csharp
using LMSupply.Embedder;

// Load the default model
await using var model = await LocalEmbedder.LoadAsync("default");

// Generate a single embedding
float[] embedding = await model.EmbedAsync("Hello, world!");
Console.WriteLine($"Embedding dimensions: {embedding.Length}");

// Generate multiple embeddings
float[][] embeddings = await model.EmbedBatchAsync(new[]
{
    "First document",
    "Second document"
});
```

## Available Models

| Alias | HuggingFace ID | Dimensions | Size | Description |
|-------|---------------|------------|------|-------------|
| `default` | sentence-transformers/all-MiniLM-L6-v2 | 384 | ~90MB | Fast, good quality |
| `large` | sentence-transformers/all-mpnet-base-v2 | 768 | ~420MB | Higher quality |
| `multilingual` | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 384 | ~470MB | 50+ languages |

You can also use any HuggingFace sentence-transformers model by its full ID:

```csharp
var model = await LocalEmbedder.LoadAsync("sentence-transformers/all-MiniLM-L12-v2");
```

## Configuration Options

```csharp
var options = new EmbedderOptions
{
    // GPU/CPU execution provider
    Provider = ExecutionProvider.Auto,  // Auto, Cpu, Cuda, DirectML, CoreML

    // Maximum sequence length (tokens)
    MaxSequenceLength = 512,

    // Normalize embeddings to unit length
    NormalizeEmbeddings = true,

    // Pooling strategy
    PoolingMode = PoolingMode.Mean,  // Mean, Cls, Max

    // Lowercase input text (for uncased models)
    DoLowerCase = true,

    // Custom cache directory
    CacheDirectory = null  // Uses ~/.cache/huggingface/hub by default
};

var model = await LocalEmbedder.LoadAsync("default", options);
```

## Similarity Calculation

```csharp
// Cosine similarity (for normalized embeddings)
float similarity = model.CosineSimilarity(embedding1, embedding2);

// Dot product
float dotProduct = model.DotProduct(embedding1, embedding2);

// Euclidean distance
float distance = model.EuclideanDistance(embedding1, embedding2);
```

## Semantic Search Example

```csharp
using LMSupply.Embedder;

await using var model = await LocalEmbedder.LoadAsync("default");

// Index documents
var documents = new[]
{
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "Python is a popular programming language",
    "Neural networks are inspired by biological neurons"
};

float[][] docEmbeddings = await model.EmbedBatchAsync(documents);

// Search
string query = "What is AI?";
float[] queryEmbedding = await model.EmbedAsync(query);

// Find most similar documents
var results = documents
    .Select((doc, i) => new
    {
        Document = doc,
        Score = model.CosineSimilarity(queryEmbedding, docEmbeddings[i])
    })
    .OrderByDescending(x => x.Score)
    .Take(3);

foreach (var result in results)
{
    Console.WriteLine($"[{result.Score:F4}] {result.Document}");
}
```

## Pooling Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `Mean` | Average of all token embeddings | General purpose (default) |
| `Cls` | Use [CLS] token embedding | Classification tasks |
| `Max` | Max pooling across tokens | Capturing key features |

```csharp
var meanOptions = new EmbedderOptions { PoolingMode = PoolingMode.Mean };
var clsOptions = new EmbedderOptions { PoolingMode = PoolingMode.Cls };
var maxOptions = new EmbedderOptions { PoolingMode = PoolingMode.Max };
```

## Thread Safety

The `IEmbeddingModel` instance is thread-safe and can be shared across multiple threads:

```csharp
await using var model = await LocalEmbedder.LoadAsync("default");

// Safe to use concurrently
await Task.WhenAll(
    Task.Run(() => model.EmbedAsync("Text 1")),
    Task.Run(() => model.EmbedAsync("Text 2")),
    Task.Run(() => model.EmbedAsync("Text 3"))
);
```

## Warmup

To avoid cold-start latency on first inference:

```csharp
await using var model = await LocalEmbedder.LoadAsync("default");
await model.WarmupAsync();  // Pre-loads the model
```

## Model Information

```csharp
var info = model.GetModelInfo();
if (info != null)
{
    Console.WriteLine($"Model: {info.DisplayName}");
    Console.WriteLine($"Dimensions: {info.EmbeddingDimensions}");
    Console.WriteLine($"Max Tokens: {info.MaxSequenceLength}");
}
```

## Local Models

You can use locally stored ONNX models:

```csharp
var model = await LocalEmbedder.LoadAsync("/path/to/model.onnx");
```

The directory should contain:
- `model.onnx` - The ONNX model file
- `tokenizer.json` or `vocab.txt` - Tokenizer configuration

## Performance Tips

1. **Reuse the model instance** - Creating a new instance loads the model from disk
2. **Use batch processing** - `EmbedBatchAsync` is more efficient than multiple `EmbedAsync` calls
3. **Enable GPU acceleration** - Install the appropriate ONNX Runtime GPU package
4. **Warmup before production** - Call `WarmupAsync()` to avoid cold-start latency
5. **Adjust batch size** for memory constraints
