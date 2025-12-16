# LMSupply.Reranker

Local semantic reranking for .NET with cross-encoder models.

## Features

- **Zero-config**: Models download automatically from HuggingFace
- **GPU Acceleration**: CUDA, DirectML (Windows), CoreML (macOS)
- **Cross-platform**: Windows, Linux, macOS
- **RAG Integration**: Perfect for improving retrieval quality

## Quick Start

```csharp
using LMSupply.Reranker;

// Load the default model
await using var reranker = await LocalReranker.LoadAsync("default");

// Rerank documents
var results = await reranker.RerankAsync(
    query: "What is machine learning?",
    documents: ["ML is a branch of AI...", "The weather is nice..."],
    topK: 5);

foreach (var result in results)
    Console.WriteLine($"{result.Index}: {result.Score:F3}");
```

## Available Models

| Alias | Model | Size | Description |
|-------|-------|------|-------------|
| `default` | MS MARCO MiniLM L6 | ~90MB | Best speed/quality balance |
| `fast` | MS MARCO TinyBERT | ~18MB | Ultra-fast, latency-critical |
| `quality` | BGE Reranker Base | ~440MB | Higher accuracy, multilingual |
| `large` | BGE Reranker Large | ~1.1GB | Highest accuracy |
| `multilingual` | BGE Reranker v2-m3 | ~1.1GB | 8K context, 100+ languages |

## GPU Acceleration

```bash
# NVIDIA GPU
dotnet add package Microsoft.ML.OnnxRuntime.Gpu

# Windows (AMD/Intel/NVIDIA)
dotnet add package Microsoft.ML.OnnxRuntime.DirectML
```
