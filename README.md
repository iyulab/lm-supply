# LocalAI

[![CI](https://github.com/iyulab/local-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/iyulab/local-ai/actions/workflows/ci.yml)
[![NuGet](https://img.shields.io/nuget/v/LocalAI.Embedder.svg)](https://www.nuget.org/packages/LocalAI.Embedder)
[![NuGet](https://img.shields.io/nuget/v/LocalAI.Reranker.svg)](https://www.nuget.org/packages/LocalAI.Reranker)
[![NuGet](https://img.shields.io/nuget/v/LocalAI.Generator.svg)](https://www.nuget.org/packages/LocalAI.Generator)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Philosophy

**Start small. Download what you need. Run locally.**

```csharp
// This is all you need. No setup. No configuration. No API keys.
await using var model = await LocalEmbedder.LoadAsync("default");
float[] embedding = await model.EmbedAsync("Hello, world!");
```

LocalAI is designed around three core principles:

### 🪶 Minimal Footprint
Your application ships with **zero bundled models**. The base package is tiny. Models, tokenizers, and runtime components are downloaded **only when first requested** and cached for reuse.

### ⚡ Lazy Everything
```
First run:  LoadAsync("default") → Downloads model → Caches → Runs inference
Next runs:  LoadAsync("default") → Uses cached model → Runs inference instantly
```
No pre-download scripts. No model management. Just use it.

### 🎯 Zero Boilerplate
Traditional approach:
```csharp
// ❌ Without LocalAI: 50+ lines of setup
var tokenizer = LoadTokenizer(modelPath);
var session = new InferenceSession(modelPath, sessionOptions);
var inputIds = tokenizer.Encode(text);
var attentionMask = CreateAttentionMask(inputIds);
var inputs = new List<NamedOnnxValue> { ... };
var outputs = session.Run(inputs);
var embeddings = PostProcess(outputs);
// ... error handling, pooling, normalization, cleanup ...
```

```csharp
// ✅ With LocalAI: 2 lines
await using var model = await LocalEmbedder.LoadAsync("default");
float[] embedding = await model.EmbedAsync("Hello, world!");
```

---

## Packages

| Package | Description | Status |
|---------|-------------|--------|
| [LocalAI.Embedder](docs/embedder.md) | Text → Vector embeddings | [![NuGet](https://img.shields.io/nuget/v/LocalAI.Embedder.svg)](https://www.nuget.org/packages/LocalAI.Embedder) |
| [LocalAI.Reranker](docs/reranker.md) | Semantic reranking for search | [![NuGet](https://img.shields.io/nuget/v/LocalAI.Reranker.svg)](https://www.nuget.org/packages/LocalAI.Reranker) |
| [LocalAI.Generator](docs/generator.md) | Text generation & chat | [![NuGet](https://img.shields.io/nuget/v/LocalAI.Generator.svg)](https://www.nuget.org/packages/LocalAI.Generator) |
| LocalAI.Transcriber | Speech → Text (Whisper) | 📋 Planned |
| LocalAI.Synthesizer | Text → Speech | 📋 Planned |
| LocalAI.Translator | Neural machine translation | 📋 Planned |
| LocalAI.Detector | Object detection | 📋 Planned |
| LocalAI.Segmenter | Image segmentation | 📋 Planned |
| LocalAI.Ocr | Document OCR | 📋 Planned |
| LocalAI.Captioner | Image → Text | 📋 Planned |

---

## Quick Start

### Text Embeddings

```csharp
using LocalAI.Embedder;

await using var model = await LocalEmbedder.LoadAsync("default");

// Single text
float[] embedding = await model.EmbedAsync("Hello, world!");

// Batch processing
float[][] embeddings = await model.EmbedBatchAsync(new[]
{
    "First document",
    "Second document",
    "Third document"
});

// Similarity
float similarity = model.CosineSimilarity(embeddings[0], embeddings[1]);
```

### Semantic Reranking

```csharp
using LocalAI.Reranker;

await using var reranker = await LocalReranker.LoadAsync("default");

var results = await reranker.RerankAsync(
    query: "What is machine learning?",
    documents: new[]
    {
        "Machine learning is a subset of artificial intelligence...",
        "The weather today is sunny and warm...",
        "Deep learning uses neural networks..."
    },
    topK: 2
);

foreach (var result in results)
{
    Console.WriteLine($"[{result.Score:F4}] {result.Document}");
}
```

### Text Generation

```csharp
using LocalAI.Generator;

// Simple generation
var generator = await TextGeneratorBuilder.Create()
    .WithDefaultModel()  // Phi-3.5 Mini
    .BuildAsync();

string response = await generator.GenerateCompleteAsync("What is machine learning?");
Console.WriteLine(response);

// Chat format
var messages = new[]
{
    new ChatMessage(ChatRole.System, "You are a helpful assistant."),
    new ChatMessage(ChatRole.User, "Explain quantum computing simply.")
};

string chatResponse = await generator.GenerateChatCompleteAsync(messages);

// Streaming
await foreach (var token in generator.GenerateAsync("Write a story:"))
{
    Console.Write(token);
}
```

---

## Available Models

### Embedder

| Alias | Model | Dimensions | Size |
|-------|-------|------------|------|
| `default` | all-MiniLM-L6-v2 | 384 | ~90MB |
| `large` | all-mpnet-base-v2 | 768 | ~420MB |
| `multilingual` | paraphrase-multilingual-MiniLM-L12-v2 | 384 | ~470MB |

### Reranker

| Alias | Model | Max Tokens | Size |
|-------|-------|------------|------|
| `default` | ms-marco-MiniLM-L-6-v2 | 512 | ~90MB |
| `quality` | ms-marco-MiniLM-L-12-v2 | 512 | ~134MB |
| `fast` | ms-marco-TinyBERT-L-2-v2 | 512 | ~18MB |
| `multilingual` | bge-reranker-v2-m3 | 8192 | ~1.1GB |

### Generator

| Alias | Model | Parameters | License |
|-------|-------|------------|---------|
| `default` | Phi-3.5-mini-instruct | 3.8B | MIT |
| `fast` | Llama-3.2-1B-Instruct | 1B | Llama 3.2 |
| `quality` | phi-4 | 14B | MIT |
| `small` | Llama-3.2-1B-Instruct | 1B | Llama 3.2 |

---

## GPU Acceleration

GPU acceleration is automatic when detected:

```csharp
// Auto-detect (default) - uses GPU if available, falls back to CPU
var options = new EmbedderOptions { Provider = ExecutionProvider.Auto };

// Force specific provider
var options = new EmbedderOptions { Provider = ExecutionProvider.Cuda };     // NVIDIA
var options = new EmbedderOptions { Provider = ExecutionProvider.DirectML }; // Windows GPU
var options = new EmbedderOptions { Provider = ExecutionProvider.CoreML };   // macOS
```

Install the appropriate ONNX Runtime package for GPU support:

```bash
dotnet add package Microsoft.ML.OnnxRuntime.Gpu       # NVIDIA CUDA
dotnet add package Microsoft.ML.OnnxRuntime.DirectML  # Windows (AMD, Intel, NVIDIA)
dotnet add package Microsoft.ML.OnnxRuntime.CoreML    # macOS
```

---

## Model Caching

Models are cached following HuggingFace Hub conventions:

- **Default**: `~/.cache/huggingface/hub`
- **Environment variables**: `HF_HUB_CACHE`, `HF_HOME`, or `XDG_CACHE_HOME`
- **Manual override**: `new EmbedderOptions { CacheDirectory = "/path/to/cache" }`

---

## Requirements

- .NET 10.0+
- Windows, Linux, or macOS

---

## Documentation

- [Embedder Guide](docs/embedder.md)
- [Reranker Guide](docs/reranker.md)
- [Generator Guide](docs/generator.md)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Release Process

Releases are automated via GitHub Actions when `Directory.Build.props` is updated:

1. Update the `<Version>` in `Directory.Build.props`
2. Commit and push to main
3. CI automatically publishes all packages to NuGet and creates a GitHub release

Requires `NUGET_API_KEY` secret configured in GitHub repository settings.
