# Model Lifecycle Guide

This guide explains how models are loaded, used, and unloaded in LMSupply.

## Overview

LMSupply follows a simple lifecycle pattern:

```
LoadAsync() → Use → DisposeAsync()
```

All model interfaces implement `IAsyncDisposable`, enabling the `await using` pattern for automatic resource cleanup.

---

## 1. Loading Models

### 1.1 Basic Loading

```csharp
// Load with model ID or alias
await using var model = await LocalEmbedder.LoadAsync("default");

// Load with "auto" for hardware-optimized model selection
await using var auto = await LocalEmbedder.LoadAsync("auto");

// Load from a local path: the same call, given the model file
await using var local = await LocalEmbedder.LoadAsync("/path/to/model.onnx");
```

### 1.2 What Happens During Load

1. **Model Resolution**: Alias → HuggingFace model ID
2. **Cache Check**: Look for the cached model in the Hugging Face cache (`HF_HUB_CACHE`, else `HF_HOME/hub`, else
   `XDG_CACHE_HOME/huggingface/hub`, else `~/.cache/huggingface/hub` — `CacheManager.GetDefaultCacheDirectory()`) — a cached file counts only when its
   length matches what the repository lists (recorded in the download manifest); a file of any other
   length is discarded and fetched again
3. **Download** (if not cached): Fetch from HuggingFace Hub — an interrupted download leaves a `.part`
   file and resumes from it on the next load
4. **ONNX Runtime Init**: Initialize inference session with GPU detection
5. **Provider Selection**: Auto-detect best execution provider (CUDA → CoreML → CPU)

### 1.3 Load Options

```csharp
var options = new EmbedderOptions
{
    // Custom cache directory
    CacheDirectory = "/custom/cache/path",

    // Force specific execution provider
    Provider = ExecutionProvider.Cuda,  // or CoreML, Cpu, Auto
};

await using var model = await LocalEmbedder.LoadAsync("default", options);
```

### 1.4 Progress Tracking

```csharp
var progress = new Progress<DownloadProgress>(p =>
{
    // A first load can also fetch a runtime (the native ONNX Runtime, llama-server) once per host
    var what = p.Kind == DownloadKind.Runtime ? "Runtime" : "Model";
    // A step that counts nothing (Preparing, Loading, a download of unknown size) has no percentage to show
    var howFar = p.IsIndeterminate ? "..." : $"{p.OverallPercentComplete:F0}%";
    Console.WriteLine($"{what} {p.FileName}: {p.Phase} {howFar}");
});

await using var model = await LocalEmbedder.LoadAsync("default", progress: progress);
```

Runtime reports carry `DownloadKind.Runtime`, the package or archive being fetched as `FileName`, and a phase
(`Preparing`, `Downloading` with byte counts, `Extracting`, `Complete`); a runtime already on disk is one `Complete`
report. Every load that takes a progress reporter forwards it to the runtime step: each `Local*.LoadAsync` (and
`LocalSegmenter.LoadInteractiveAsync`), `LocalGenerator.LoadAsync` and `LoadFromPathAsync` (a model already on disk can
still need its runtime), and `TextGeneratorBuilder.WithDownloadProgress`. `LocalImageGenerator.LoadAsync` takes the
same `IProgress<DownloadProgress>` as the others.

What a load reports, in order:

| Phase | When | Counts |
|---|---|---|
| `Preparing` | Before a file or runtime is looked up | nothing (`IsIndeterminate`) |
| `Downloading` | Bytes are moving. A file found usable in the cache is never reported as a download | bytes of the file, and of the whole download when every size is known |
| `Extracting`, `Verifying`, `Finalizing` | An archive is unpacked (a runtime, on first use), content is checked, a file moves into the cache | bytes when the step has a total |
| `Complete` | The file is in place — downloaded, or found in the cache. Once per file; a cached file carries its size | `PercentComplete` is 100 |
| `Loading` | The files are in place and the model is being brought into memory: llama-server starts and reads the weights, or an ONNX Runtime session is created. Reported for each file loaded, under the name its download had (`unet/model.onnx`) | nothing (`IsIndeterminate`) |

`Loading` is the last report of a file. Nothing is reported when loading ends: the load is over when `LoadAsync`
returns. A GGUF load that shares a llama-server already running for the same model loads nothing and reports no
`Loading`. The byte fields are bytes or zero — never a percentage in disguise — so `IsIndeterminate` is the one test
for «show this step as busy»: `PercentComplete` and `OverallPercentComplete` are zero for such a report, and that zero
means «not measured».

A load whose model is cached reports, for a GGUF model:

```
Model   Qwen3-8B-Q4_K_M.gguf  Complete   (the file's size)
Runtime llama-server          Preparing
Model   Qwen3-8B-Q4_K_M.gguf  Loading
```

---

## 2. Using Models

### 2.1 First Inference Latency

The first inference call may be slower due to:
- JIT compilation of ONNX operators
- GPU kernel compilation (CUDA)
- Memory allocation

### 2.2 Warmup Pattern

Use `WarmupAsync()` to pre-compile before production use:

```csharp
await using var model = await LocalEmbedder.LoadAsync("default");

// Warm up the model (runs a dummy inference)
await model.WarmupAsync();

// Subsequent calls will be faster
var embedding = await model.EmbedAsync("Hello world");
```

**When to use warmup:**
- Server applications where latency matters
- Before processing large batches
- After loading model in background

### 2.3 Thread Safety

All LMSupply models are **thread-safe** for inference:

```csharp
await using var model = await LocalEmbedder.LoadAsync("default");

// Safe to call from multiple threads
var tasks = texts.Select(text => model.EmbedAsync(text).AsTask());
var embeddings = await Task.WhenAll(tasks);
```

---

## 3. Unloading Models

### 3.1 Automatic Cleanup with `await using`

```csharp
await using var model = await LocalEmbedder.LoadAsync("default");
// Use model...
// Automatically disposed at end of scope
```

### 3.2 Manual Disposal

```csharp
var model = await LocalEmbedder.LoadAsync("default");
try
{
    // Use model...
}
finally
{
    await model.DisposeAsync();
}
```

### 3.3 What Happens During Dispose

1. **ONNX Session Release**: Native resources freed
2. **GPU Memory Release**: VRAM returned to system
3. **Tokenizer Cleanup**: Managed resources released

---

## 4. Multiple Models

### 4.1 Loading Multiple Models

Each `LoadAsync()` creates an independent model instance:

```csharp
await using var embedder = await LocalEmbedder.LoadAsync("default");
await using var reranker = await LocalReranker.LoadAsync("default");

// Both models active simultaneously
var embeddings = await embedder.EmbedAsync(texts);
var ranked = await reranker.RerankAsync(query, documents);
```

### 4.2 Memory Considerations

- Each model consumes GPU VRAM independently
- Monitor total VRAM usage when loading multiple models
- Consider sequential loading if VRAM is limited

```csharp
// Sequential approach for limited VRAM
float[][] embeddings;
IReadOnlyList<RankedResult> ranked;

await using (var embedder = await LocalEmbedder.LoadAsync("default"))
{
    embeddings = await embedder.EmbedAsync(texts);
}
// Embedder released, VRAM freed

await using (var reranker = await LocalReranker.LoadAsync("quality"))
{
    ranked = await reranker.RerankAsync(query, documents);
}
```

---

## 5. Model Information

### 5.1 Querying Model State

```csharp
await using var model = await LocalEmbedder.LoadAsync("default");

var info = model.GetModelInfo();   // null for a model the catalog does not describe
Console.WriteLine($"Model: {model.ModelId}");
Console.WriteLine($"Repository: {info?.RepoId}");
Console.WriteLine($"Dimensions: {model.Dimensions}");
```

### 5.2 Runtime Diagnostics

Check GPU activation status:

```csharp
// Every model reports the execution providers its session uses
Console.WriteLine($"GPU Active: {model.IsGpuActive}");
Console.WriteLine($"Providers: {string.Join(", ", model.ActiveProviders)}");
```

---

## 6. Best Practices

### 6.1 Long-Running Applications

Keep models loaded for the application lifetime:

```csharp
public class EmbeddingService : IAsyncDisposable
{
    private readonly IEmbeddingModel _model;

    public static async Task<EmbeddingService> CreateAsync()
    {
        var model = await LocalEmbedder.LoadAsync("auto");
        await model.WarmupAsync();
        return new EmbeddingService(model);
    }

    private EmbeddingService(IEmbeddingModel model) => _model = model;

    public ValueTask<float[]> EmbedAsync(string text, CancellationToken cancellationToken = default) =>
        _model.EmbedAsync(text, cancellationToken);

    public ValueTask DisposeAsync() => _model.DisposeAsync();
}
```

### 6.2 Batch Processing

Use batch methods for efficiency:

```csharp
// Efficient: Single batch call
var embeddings = await model.EmbedAsync(texts);

// Less efficient: Multiple individual calls
foreach (var text in texts)
{
    var embedding = await model.EmbedAsync(text);
}
```

### 6.3 Error Handling

```csharp
try
{
    await using var model = await LocalEmbedder.LoadAsync("default");
    var result = await model.EmbedAsync(text);
}
catch (ModelNotFoundException ex)
{
    // Model not found in registry or HuggingFace
    Console.WriteLine($"Model not found: {ex.ModelId}");
}
catch (ModelLoadException ex)
{
    // Failed to load model (corrupt files, missing dependencies)
    Console.WriteLine($"Load failed: {ex.Message}");
}
catch (InferenceException ex)
{
    // Runtime inference error
    Console.WriteLine($"Inference failed: {ex.Message}");
}
```

---

## 7. Summary

| Phase | Action | Notes |
|-------|--------|-------|
| Load | `LoadAsync()` | Downloads if needed, initializes GPU |
| Warmup | `WarmupAsync()` | Optional, reduces first-call latency |
| Use | Domain methods | Thread-safe, batch-optimized |
| Unload | `DisposeAsync()` | Use `await using` for automatic cleanup |

**Key Points:**
- Always use `await using` for automatic resource cleanup
- Use `"auto"` alias for hardware-optimized model selection
- Call `WarmupAsync()` in latency-sensitive applications
- Models are thread-safe for concurrent inference
