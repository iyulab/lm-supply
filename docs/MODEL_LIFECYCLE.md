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
await using var model = await LocalEmbedder.LoadAsync("auto");

// Load from local path
await using var model = await LocalEmbedder.LoadFromPathAsync("/path/to/model");
```

### 1.2 What Happens During Load

1. **Model Resolution**: Alias → HuggingFace model ID
2. **Cache Check**: Look for cached model in `~/.cache/lm-supply/` — a cached file counts only when its
   length matches what the repository lists (recorded in the download manifest); a file of any other
   length is discarded and fetched again
3. **Download** (if not cached): Fetch from HuggingFace Hub — an interrupted download leaves a `.part`
   file and resumes from it on the next load
4. **ONNX Runtime Init**: Initialize inference session with GPU detection
5. **Provider Selection**: Auto-detect best execution provider (CUDA → DirectML → CoreML → CPU)

### 1.3 Load Options

```csharp
var options = new EmbedderOptions
{
    // Custom cache directory
    CacheDirectory = "/custom/cache/path",

    // Force specific execution provider
    Provider = ExecutionProvider.Cuda,  // or DirectML, CoreML, Cpu, Auto
};

await using var model = await LocalEmbedder.LoadAsync("default", options);
```

### 1.4 Progress Tracking

```csharp
var progress = new Progress<DownloadProgress>(p =>
{
    Console.WriteLine($"Downloading: {p.OverallPercentComplete:F0}%");
});

await using var model = await LocalEmbedder.LoadAsync("default", progress: progress);
```

---

## 2. Using Models

### 2.1 First Inference Latency

The first inference call may be slower due to:
- JIT compilation of ONNX operators
- GPU kernel compilation (CUDA/DirectML)
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
var tasks = texts.Select(text => model.EmbedAsync(text));
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

var info = model.GetModelInfo();
Console.WriteLine($"Model: {info.ModelId}");
Console.WriteLine($"Path: {info.ModelPath}");
Console.WriteLine($"Dimensions: {info.Dimensions}");
```

### 5.2 Runtime Diagnostics

Check GPU activation status:

```csharp
// For Transcriber (currently)
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

    public Task<float[]> EmbedAsync(string text) => _model.EmbedAsync(text);

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
