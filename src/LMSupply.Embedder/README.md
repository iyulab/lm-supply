# LMSupply.Embedder

Local text embedding for .NET with automatic model downloading.

## Features

- **Zero-config**: Models download automatically from HuggingFace
- **GPU Acceleration**: CUDA, DirectML (Windows), CoreML (macOS)
- **Cross-platform**: Windows, Linux, macOS
- **Simple API**: Just 2 lines of code to get started

## Quick Start

```csharp
using LMSupply.Embedder;

// Load the default model
await using var model = await LocalEmbedder.LoadAsync("default");

// Generate embeddings
float[] embedding = await model.EmbedAsync("Hello, world!");
Console.WriteLine($"Dimensions: {embedding.Length}");
```

## Query/Passage Embeddings

Some models (the E5 family, Nomic) are fine-tuned with an asymmetric text-prefix convention —
query embeddings and document/passage embeddings need different prefixes for accurate retrieval.
`EmbedQueryAsync`/`EmbedPassageAsync` apply the right prefix automatically from the model's
`ModelInfo` (a no-op passthrough for models that don't need one):

```csharp
await using var model = await LocalEmbedder.LoadAsync("multilingual-e5-base");

float[] queryEmbedding = await model.EmbedQueryAsync("what is the capital of France?");
float[] passageEmbedding = await model.EmbedPassageAsync("Paris is the capital of France.");
```

Batch and Matryoshka-truncated (`dimensions:`) overloads exist for both, mirroring `EmbedAsync`.

## Available Models

| Alias | Model | Dimensions | Description |
|-------|-------|------------|-------------|
| `default` | BGE-Small-EN-v1.5 | 384 | Best balance of speed and quality |
| `fast` | all-MiniLM-L6-v2 | 384 | Ultra-lightweight, fastest |
| `quality` | BGE-Base-EN-v1.5 | 768 | Higher accuracy |
| `large` | Nomic-Embed-v1.5 | 768 | 8K context, top performer |
| `multilingual` | E5-Base | 768 | 100+ languages |

## GPU Acceleration

```bash
# NVIDIA GPU
dotnet add package Microsoft.ML.OnnxRuntime.Gpu

# Windows (AMD/Intel/NVIDIA)
dotnet add package Microsoft.ML.OnnxRuntime.DirectML
```
