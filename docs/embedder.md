# LMSupply.Embedder

A lightweight, zero-configuration text embedding library for .NET with automatic GPU acceleration.

Supports both **ONNX** models (sentence-transformers) and **GGUF** models (via llama-server).

## Installation

```bash
dotnet add package LMSupply.Embedder
```

GPU acceleration is **automatic** — LMSupply detects your hardware and downloads appropriate runtime binaries on first use. No additional packages required.

## Basic Usage

```csharp
using LMSupply.Embedder;

// Load using alias (recommended)
await using var model = await LocalEmbedder.LoadAsync("default");

// Or use "auto" for hardware-optimized model selection
await using var model = await LocalEmbedder.LoadAsync("auto");

// Or load directly from HuggingFace with owner/repo-name format
await using var model = await LocalEmbedder.LoadAsync("BAAI/bge-large-en-v1.5");

// Generate a single embedding
float[] embedding = await model.EmbedAsync("Hello, world!");
Console.WriteLine($"Embedding dimensions: {embedding.Length}");

// Generate multiple embeddings
float[][] embeddings = await model.EmbedAsync(new[]
{
    "First document",
    "Second document"
});
```

## Available Models (ONNX)

> **v0.34+ change:** `default` now resolves to **BGE-M3** (1024-dim, 8K context, 100+ languages).
> The former `multilingual` alias has been removed — use `default` instead.

| Alias | Model | Dimensions | Params | Context | Best For |
|-------|-------|------------|--------|---------|----------|
| `auto` | Hardware-optimized | varies | varies | varies | Auto-select by available VRAM |
| `default` | bge-m3 | 1024 | 568M | 8192 | SOTA multilingual, 100+ langs, dense+sparse (v0.34+) |
| `quality` | bge-m3 | 1024 | 568M | 8192 | Same as `default`; for pipelines that pin quality tier |
| `fast` | multilingual-e5-small | 384 | 118M | 512 | Lightweight, 100+ langs |
| `large` | multilingual-e5-large | 1024 | 560M | 512 | Highest dense quality, 100+ langs |

### Using HuggingFace Repository ID

You can load any HuggingFace ONNX embedding model directly with `owner/repo-name` format:

```csharp
// Load by HuggingFace repository ID
await using var model = await LocalEmbedder.LoadAsync("BAAI/bge-large-en-v1.5");
await using var model = await LocalEmbedder.LoadAsync("sentence-transformers/all-MiniLM-L12-v2");
await using var model = await LocalEmbedder.LoadAsync("intfloat/multilingual-e5-large");
```

The system automatically discovers the ONNX files in the repository.

## GGUF Models (via llama-server)

GGUF embedding models are auto-detected by repo name patterns (`-GGUF`, `_gguf`) or `.gguf` extension. LMSupply automatically downloads and manages llama-server binaries for GPU-accelerated inference.

When a GGUF repository contains multiple quantization files, LMSupply selects the **largest quantization that fits in available memory** (VRAM + RAM). Only single-file GGUF models are supported for embedding (split multi-part files are excluded).

```csharp
using LMSupply.Embedder;

// Load GGUF model (auto-detected by "-GGUF" in repo name)
await using var model = await LocalEmbedder.LoadAsync("nomic-ai/nomic-embed-text-v1.5-GGUF");

// Usage is identical to ONNX models
float[] embedding = await model.EmbedAsync("Hello from GGUF!");

// Batch processing
float[][] embeddings = await model.EmbedAsync(new[]
{
    "First document",
    "Second document"
});
```

### Available GGUF Embedding Models

| Model Repository | Dims | Context | Best For |
|------------------|------|---------|----------|
| `nomic-ai/nomic-embed-text-v1.5-GGUF` | 768 | 8K | Long context, matryoshka |
| `BAAI/bge-small-en-v1.5-GGUF` | 384 | 512 | Compact and fast |
| `BAAI/bge-base-en-v1.5-GGUF` | 768 | 512 | Quality balance |

You can also use local GGUF files:

```csharp
await using var model = await LocalEmbedder.LoadAsync("/path/to/embedding-model.gguf");
```

## Multilingual Support

`default` (BGE-M3) supports 100+ languages with 1024-dim embeddings and 8K context. It is the recommended model for all new projects:

```csharp
// default = BGE-M3 since v0.34 (the former "multilingual" alias has been removed)
await using var model = await LocalEmbedder.LoadAsync("default");

// Korean text embedding
float[] koreanEmbedding = await model.EmbedAsync("안녕하세요, 세계!");

// Japanese text embedding
float[] japaneseEmbedding = await model.EmbedAsync("こんにちは、世界！");

// Chinese text embedding
float[] chineseEmbedding = await model.EmbedAsync("你好，世界！");

// Cross-lingual similarity works!
float similarity = LocalEmbedder.CosineSimilarity(koreanEmbedding, japaneseEmbedding);
```

BGE-M3 produces 1024-dimensional embeddings and supports passages up to 8192 tokens, making it ideal for long multilingual documents.

## Configuration Options

```csharp
var options = new EmbedderOptions
{
    // GPU/CPU execution provider
    Provider = ExecutionProvider.Auto,  // Auto, Cpu, Cuda, CoreML

    // Maximum sequence length (tokens). Left null (the default), the model decides: its
    // sentence_bert_config.json max_seq_length first, then the catalog, then 512. A value is used
    // as given. After loading, this property holds the length in effect.
    MaxSequenceLength = null,

    // Normalize embeddings to unit length
    NormalizeEmbeddings = true,

    // Pooling strategy (ONNX models only). Left null (the default), the model decides: its
    // 1_Pooling/config.json first, then the catalog, then Mean. A value is used as given.
    PoolingMode = null,  // Mean, Cls, Max

    // Lowercase input text (for uncased models)
    DoLowerCase = true,

    // CPU threads for inference. null (the default) = ONNX Runtime's pool: a thread per physical core,
    // spin-waiting between runs for throughput. A number = that many threads, and they stop
    // spin-waiting, so an interactive app that embeds a field at a time leaves idle cores idle.
    ThreadCount = null,

    // Custom cache directory
    CacheDirectory = null,  // Uses ~/.cache/huggingface/hub by default

    // true: load only from the cache, throw ModelNotFoundException if a file is missing, write nothing
    DisableAutoDownload = false
};

var model = await LocalEmbedder.LoadAsync("default", options);
```

## Similarity Calculation

```csharp
// Cosine similarity (for normalized embeddings)
float similarity = LocalEmbedder.CosineSimilarity(embedding1, embedding2);

// Dot product
float dotProduct = LocalEmbedder.DotProduct(embedding1, embedding2);

// Euclidean distance
float distance = LocalEmbedder.EuclideanDistance(embedding1, embedding2);
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

// Each side of retrieval takes the model's own prefix (a no-op for models without one)
float[][] docEmbeddings = await model.EmbedPassageAsync(documents);

// Search
string query = "What is AI?";
float[] queryEmbedding = await model.EmbedQueryAsync(query);

// Find most similar documents
var results = documents
    .Select((doc, i) => new
    {
        Document = doc,
        Score = LocalEmbedder.CosineSimilarity(queryEmbedding, docEmbeddings[i])
    })
    .OrderByDescending(x => x.Score)
    .Take(3);

foreach (var result in results)
{
    Console.WriteLine($"[{result.Score:F4}] {result.Document}");
}
```

## Clustering Example

```csharp
using LMSupply.Embedder;

await using var model = await LocalEmbedder.LoadAsync("default");

var texts = new[]
{
    // Technology
    "Artificial intelligence is changing industries.",
    "Machine learning models improve with more data.",
    "Cloud computing enables scalable applications.",
    // Nature
    "The forest is home to many species of birds.",
    "Rivers flow from mountains to the sea.",
    // Food
    "Italian pasta is often served with tomato sauce.",
    "Sushi is a traditional Japanese dish."
};

var embeddings = await model.EmbedAsync(texts);

// Find texts with similarity > 0.5
for (int i = 0; i < texts.Length; i++)
{
    for (int j = i + 1; j < texts.Length; j++)
    {
        var sim = LocalEmbedder.CosineSimilarity(embeddings[i], embeddings[j]);
        if (sim > 0.5)
        {
            Console.WriteLine($"Similar: \"{texts[i]}\" <-> \"{texts[j]}\" ({sim:F3})");
        }
    }
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
await Parallel.ForEachAsync(documents, async (doc, ct) =>
{
    var embedding = await model.EmbedAsync(doc, ct);
    // Process embedding...
});
```

## Warmup

To avoid cold-start latency on first inference:

```csharp
await using var model = await LocalEmbedder.LoadAsync("default");
await model.WarmupAsync();  // Pre-loads the model
```

## Cancellation & Timeouts

All `EmbedAsync` overloads honor the supplied `CancellationToken`. Because ONNX inference is a
blocking native call, the model enforces two guarantees:

1. **Control return (guaranteed).** When the token is cancelled, the `await` returns control to the
   caller within bound and throws `OperationCanceledException` — even if a native call (for example
   a cold GPU kernel initialization) is still blocked internally.
2. **Native termination (best-effort).** The cancelled token also asks the running ONNX graph to
   terminate cooperatively (`RunOptions.Terminate`). This is honored *between operators*, so it
   frees the worker thread in most cases but cannot preempt a hang inside a single kernel.

Encode the deadline on the token you pass — the library does not impose a default timeout:

```csharp
using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));

// On a cold-init hang, control returns within ~10s instead of blocking forever.
float[] embedding = await model.EmbedAsync("Hello, world!", cts.Token);
```

> Consumers no longer need to wrap `EmbedAsync` in their own `WaitAsync(timeout, ct)` guard — pass a
> `CancellationTokenSource` with `CancelAfter`/a timeout and the call returns control on its own.

## Model Information

```csharp
var info = model.GetModelInfo();
if (info != null)
{
    Console.WriteLine($"Model: {info.RepoId}");
    Console.WriteLine($"Dimensions: {info.Dimensions}");
    Console.WriteLine($"Max Tokens: {info.MaxSequenceLength}");
}
```

### Vector-space revision

A release can change the vectors a model id produces — a tokenizer fix, a pooling read from the
model's own files, a different sequence length — and every vector stored before it is then stale.
`IEmbeddingModel.VectorSpaceRevision` is an opaque string derived from what the loader actually
did for this model (tokenizer and normalization convention, pooling, L2 normalization, query/passage/default
prefixes, effective sequence length, the model file opened, and the epoch of this library's
implementation of each step). Store it next to the vectors; on a later load, a different value means
those vectors need re-embedding for this model — and only for this model, since a fix to one
tokenizer family moves that family alone.

```csharp
await using var model = await LocalEmbedder.LoadAsync("default");
var revision = model.VectorSpaceRevision;   // e.g. "c586ab6fab5393a1"

// The same value before the model is loaded, from the cached files alone (0.72.0) — no session,
// no download; null when it cannot be known without loading (not cached, no declared dimension, GGUF).
string? early = await LocalEmbedder.GetVectorSpaceRevisionAsync("default");

// On load, compare with what the index was built under:
if (index.EmbeddingRevision != revision)
    await index.ReembedAsync(model);        // stale for this model id
```

**Retrieval revision.** A store that embeds documents with `EmbedPassageAsync` and searches with `EmbedQueryAsync`
never calls `EmbedAsync`, so the prefix `EmbedAsync` applies to a model with a default prefix (the E5 family, Nomic) does
not shape its vectors. `IEmbeddingModel.RetrievalVectorSpaceRevision` (and `LocalEmbedder.GetRetrievalVectorSpaceRevisionAsync`
before the load) is the same value without that prefix: key a retrieval index on it, and a change to the default prefix
does not ask it to re-embed. For a model without a default prefix the two values are equal, and the retrieval value of
every model equals the `VectorSpaceRevision` it had before default prefixes existed (0.111.0).

```csharp
var retrieval = model.RetrievalVectorSpaceRevision;
string? earlyRetrieval = await LocalEmbedder.GetRetrievalVectorSpaceRevisionAsync("fast");
```

The execution provider and GPU are not part of the value (they change floating-point noise, not the
space); for a GGUF model the llama-server binary version is not either. The value is a hash; the line
it was computed from is traced at load as `[LocalEmbedder.vectorspace]` when two revisions need to be
compared by eye.

## Local Models

You can use locally stored ONNX models:

```csharp
var model = await LocalEmbedder.LoadAsync("/path/to/model.onnx");
```

The model's directory (or, for a copied repository with the model in `onnx/`, the repository root one level up)
should contain:
- `model.onnx` - the ONNX model file
- a tokenizer: `tokenizer.json`, `vocab.txt` (WordPiece) or `sentencepiece.bpe.model` (SentencePiece — the
  multilingual models)

Root files of a sentence-transformers repository (`modules.json`, `1_Pooling/`, `config_sentence_transformers.json`,
`sentence_bert_config.json`) are read from the repository root, as a load by repository id reads them.

A model the catalog knows loads with the catalog's declarations — query/passage prefixes, pooling and sequence length —
when the files say which repository they are: the download manifest LMSupply writes next to a model it fetched, or
`_name_or_path` in `config.json`. A copied multilingual E5 model therefore keeps its `query: `/`passage: ` prefixes,
which its repository does not declare anywhere. Prompts the files declare themselves
(`config_sentence_transformers.json` `prompts`) still win over the catalog's, so a fine-tune whose config names its base
model keeps its own. Shipping a model for offline use through the catalog id with
`CacheDirectory` + `DisableAutoDownload` (README «Offline / air-gapped use») works the same way.

## Download Size and Cache State

`LocalEmbedder.GetDownloadSizeBytesAsync(modelIdOrPath, options)` is the number of bytes a first load would fetch: the
files that load picks on this host, at the lengths the repository lists. It reads the repository listing (cached, and
reused by the download that follows) and downloads nothing. The figure is the whole download whatever the cache already
holds, and `IsModelDownloaded` answers what is present. A local path is 0. Runtimes a first load also provisions (the
native ONNX Runtime, or llama-server for GGUF) are not counted.

`LocalEmbedder.GetRemainingDownloadBytesAsync(modelIdOrPath, options)` is what that load would still fetch now: the same
files, less those the cache already holds at the listed length. It is 0 once the model is cached, and the missing part
of a partly cached model otherwise — the figure a consent screen asks about. A partly downloaded file counts in full.
Every domain that has `GetDownloadSizeBytesAsync` has it (Reranker, Generator, Captioner, OCR, Transcriber).

`LocalEmbedder.Describe(modelIdOrPath)` names what that load would open — repository, backend (`ModelBackend.Onnx` /
`ModelBackend.Gguf`), display name and curated licence — without downloading anything. A user alias is followed; a GGUF
repository named `<org>/<model>-GGUF` whose source model is in the catalog reports that model's licence; any other
repository or path reports its own name and a null licence.

## Microsoft.Extensions.AI

`model.AsEmbeddingGenerator(textKind)` returns an `IEmbeddingGenerator<string, Embedding<float>>` over a loaded model:

| Behaviour | |
|---|---|
| `EmbeddingTextKind.Default` (default) | the model's `DefaultPrefix` (`EmbedAsync`) — similarity, clustering, features |
| `EmbeddingTextKind.Query` / `Passage` | the model's `QueryPrefix` / `PassagePrefix` (`EmbedQueryAsync` / `EmbedPassageAsync`) |
| `EmbeddingTextKind.Raw` | texts as given (`EmbedRawAsync`) |
| `EmbeddingGenerationOptions.Dimensions` | Matryoshka truncation, 1 to `model.Dimensions`, otherwise `ArgumentOutOfRangeException` |
| `EmbeddingGenerationOptions.ModelId` | must be this model's id (any case), otherwise `ArgumentException` |
| `Metadata` | provider `"LMSupply"`, the model id, the native dimension count |
| `GetService(typeof(IEmbeddingModel))` | the wrapped model |
| `Dispose()` | nothing: the model belongs to whoever loaded it |

## Performance Tips

1. **Reuse the model instance** - Creating a new instance loads the model from disk
2. **Use batch processing** - `EmbedAsync(string[])` is more efficient than multiple `EmbedAsync` calls
3. **Enable GPU acceleration** - LMSupply automatically uses GPU when available
4. **Warmup before production** - Call `WarmupAsync()` to avoid cold-start latency
5. **Choose the right model** - Use `fast` for latency-sensitive, `quality` for accuracy

## GGUF vs ONNX

| Feature | ONNX | GGUF |
|---------|------|------|
| Format | ONNX Runtime | llama-server |
| GPU Support | CUDA, CoreML | CUDA, Metal, Vulkan |
| Quantization | FP32/FP16/INT8 | Q4/Q5/Q8/F16 |
| Model Sources | HuggingFace ONNX repos | HuggingFace GGUF repos |
| Best For | Standard transformers | Long context, quantized models |
| Server Mode | In-process | HTTP server (pooled) |
