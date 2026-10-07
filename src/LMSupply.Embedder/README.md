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

Some models (the E5 family, Nomic) are fine-tuned with a text-prefix convention — query embeddings
and document/passage embeddings need different prefixes for accurate retrieval, and text with no
retrieval role (similarity, clustering) takes a third. Every entry point applies the model's own
convention from its `ModelInfo` (a no-op for models that don't need one):

| Method | Prefix applied | E5 | Nomic |
|---|---|---|---|
| `EmbedAsync` | `DefaultPrefix` — similarity, clustering, features | `query: ` | `clustering: ` |
| `EmbedQueryAsync` | `QueryPrefix` — the search side of retrieval | `query: ` | `search_query: ` |
| `EmbedPassageAsync` | `PassagePrefix` — what retrieval searches | `passage: ` | `search_document: ` |
| `EmbedRawAsync` | none — the text as given, for text that already carries its instruction | | |

```csharp
await using var model = await LocalEmbedder.LoadAsync("multilingual-e5-base");

float[] queryEmbedding = await model.EmbedQueryAsync("what is the capital of France?");
float[] passageEmbedding = await model.EmbedPassageAsync("Paris is the capital of France.");
```

Batch and Matryoshka-truncated (`dimensions:`) overloads exist for all four. A model loaded by
repository id takes these from `config_sentence_transformers.json` (`prompts` and
`default_prompt_name`), as sentence-transformers does.

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
and `EmbedAsync`/`EmbedQueryAsync`/`EmbedPassageAsync` embed its text as given.

## GPU Acceleration

Do not add ONNX Runtime packages (`Microsoft.ML.OnnxRuntime*`): LMSupply provisions the runtime itself, and a
second copy conflicts with it. `ExecutionProvider.Auto` uses CUDA when the CUDA 12 runtime and cuDNN 9 are
installed on the machine, CoreML on macOS, and the CPU otherwise. On Windows with an AMD or Intel GPU, ONNX
sessions run on the CPU (DirectML was removed in 0.67.0). See
[GPU acceleration](https://github.com/iyulab/lm-supply#gpu-acceleration).
