# LocalAI.Embedder

Local text embedding for .NET with automatic model downloading.

## Features

- **Zero-config**: Models download automatically from HuggingFace
- **GPU Acceleration**: CUDA, DirectML (Windows), CoreML (macOS)
- **Cross-platform**: Windows, Linux, macOS
- **Simple API**: Just 2 lines of code to get started

## Quick Start

```csharp
using LocalAI.Embedder;

// Load the default model
await using var model = await LocalEmbedder.LoadAsync("default");

// Generate embeddings
float[] embedding = await model.EmbedAsync("Hello, world!");
Console.WriteLine($"Dimensions: {embedding.Length}");
```

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
