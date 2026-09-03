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

Four standard aliases (`LocalEmbedder.LoadAsync("default")`, etc.) plus a longer list of models
loadable by their explicit short name (`LocalEmbedder.LoadAsync("multilingual-e5-base")`).

### Aliases

| Alias | Model | Dimensions | Prefix | Description |
|-------|-------|------------|--------|-------------|
| `default` | BAAI/bge-m3 | 1024 | — | 568M params, 100+ languages, 8K context, SOTA multilingual |
| `fast` | intfloat/multilingual-e5-small | 384 | query/passage | 118M params, 100+ languages, lightweight |
| `quality` | BAAI/bge-m3 | 1024 | — | Same model as `default`; exposed separately for pipelines that explicitly request the quality tier |
| `large` | intfloat/multilingual-e5-large | 1024 | query/passage | 560M params, 100+ languages, highest dense quality (512-token context limit — use `default` for long documents) |

### Explicit models (by short name)

| Model | Dimensions | Prefix | Description |
|-------|------------|--------|-------------|
| `nomic-embed-text-v1.5` | 768 (Matryoshka 64–768) | search_query/search_document | 137M params, English-first, 8K context |
| `all-mpnet-base-v2` | 768 | — | 110M params, legacy quality model, English |
| `bge-base-en-v1.5` | 768 | — | 110M params, excellent quality, English |
| `bge-large-en-v1.5` | 1024 | — | 335M params, highest accuracy BGE, English |
| `e5-small-v2` | 384 | query/passage | 33M params, English |
| `e5-base-v2` | 768 | query/passage | 110M params, excellent retrieval, English |
| `multilingual-e5-small` | 384 | query/passage | 118M params, 100+ languages, compact |
| `multilingual-e5-base` | 768 | query/passage | 278M params, 100+ languages, quality |
| `multilingual-e5-large` | 1024 | query/passage | 560M params, 100+ languages, highest quality |
| `gte-large-en-v1.5` | 1024 | — | 434M params, 8K context, highest accuracy GTE |

"Prefix" marks models fine-tuned with the query/passage convention (see
[Query/Passage Embeddings](#querypassage-embeddings) above) — `—` means the model needs no prefix
and `EmbedQueryAsync`/`EmbedPassageAsync` behave as a plain passthrough for it.

## GPU Acceleration

```bash
# NVIDIA GPU
dotnet add package Microsoft.ML.OnnxRuntime.Gpu

# Windows (AMD/Intel/NVIDIA)
dotnet add package Microsoft.ML.OnnxRuntime.DirectML
```
