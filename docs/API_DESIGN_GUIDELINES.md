# LocalAI API Design Guidelines

This document defines the API design principles and patterns for all LocalAI packages.
These guidelines ensure a consistent developer experience across the LocalAI family while allowing each package to express its unique characteristics.

## Design Priorities

1. **Domain-First**: Each package's API should optimally serve its specific use case
2. **Consistency**: Predictable patterns across the LocalAI family
3. **Simplicity**: Minimal boilerplate for common operations
4. **Flexibility**: Advanced options available without cluttering the simple path

---

## 1. Package Naming Convention

```
LocalAI.{Domain}
```

| Package | Domain | Description |
|---------|--------|-------------|
| LocalAI.Embedder | Text → Vector | Embedding generation |
| LocalAI.Reranker | Query + Docs → Scores | Semantic reranking |
| LocalAI.Generator | Prompt → Text | Text generation |
| LocalAI.Transcriber | Audio → Text | Speech recognition |
| LocalAI.Synthesizer | Text → Audio | Speech synthesis |
| LocalAI.Captioner | Image → Text | Image captioning |
| LocalAI.Detector | Image → Boxes | Object detection |

---

## 2. Entry Point Pattern

### 2.1 Static Factory Class

Each package MUST provide a static factory class as the primary entry point:

```csharp
// Pattern: Local{Domain}
public static class Local{Domain}
{
    // Primary factory method
    public static Task<I{Domain}Model> LoadAsync(
        string modelIdOrPath,
        {Domain}Options? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default);

    // Default model shortcut
    public static Task<I{Domain}Model> LoadDefaultAsync(
        {Domain}Options? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default);
}
```

**Examples:**
```csharp
// Embedder
var embedder = await LocalEmbedder.LoadAsync("default");
var embedder = await LocalEmbedder.LoadAsync("bge-small-en-v1.5");

// Reranker
var reranker = await LocalReranker.LoadAsync("default");

// Generator
var generator = await LocalGenerator.LoadAsync("default");

// Future: Transcriber
var transcriber = await LocalTranscriber.LoadAsync("whisper-base");
```

### 2.2 Builder Pattern (Optional, for Complex Configuration)

Packages with complex configuration MAY provide a fluent builder:

```csharp
// Pattern: {Domain}Builder
public sealed class {Domain}Builder
{
    public static {Domain}Builder Create() => new();

    // Model selection
    public {Domain}Builder WithModel(string modelId);
    public {Domain}Builder WithModelPath(string path);
    public {Domain}Builder WithDefaultModel();
    public {Domain}Builder WithModel({Domain}ModelPreset preset);

    // Common configuration
    public {Domain}Builder WithProvider(ExecutionProvider provider);
    public {Domain}Builder WithCacheDirectory(string path);

    // Domain-specific configuration
    public {Domain}Builder With{DomainSpecific}(...);

    // Build
    public Task<I{Domain}Model> BuildAsync(CancellationToken cancellationToken = default);
}
```

**When to provide a Builder:**
- Many configuration options (>5)
- Domain-specific presets (Creative, Precise, etc.)
- Complex initialization flows
- Pooling or memory management features

---

## 3. Model Interface Pattern

### 3.1 Interface Naming

```csharp
// Pattern: I{Domain}Model
public interface I{Domain}Model : IAsyncDisposable
```

| Package | Interface |
|---------|-----------|
| Embedder | `IEmbeddingModel` |
| Reranker | `IRerankerModel` |
| Generator | `IGeneratorModel` |
| Transcriber | `ITranscriberModel` |

### 3.2 Required Members

Every model interface MUST include:

```csharp
public interface I{Domain}Model : IAsyncDisposable
{
    /// <summary>
    /// Gets the model identifier.
    /// </summary>
    string ModelId { get; }

    /// <summary>
    /// Pre-loads the model to avoid cold start latency.
    /// </summary>
    Task WarmupAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets information about the loaded model.
    /// </summary>
    {Domain}ModelInfo? GetModelInfo();
}
```

### 3.3 Domain-Specific Methods

Each interface defines its core operations following the pattern:

```csharp
// Synchronous single-item (if applicable)
{Output} {Verb}({Input} input);

// Async single-item
Task<{Output}> {Verb}Async({Input} input, CancellationToken cancellationToken = default);
// or
ValueTask<{Output}> {Verb}Async({Input} input, CancellationToken cancellationToken = default);

// Async batch
Task<{Output}[]> {Verb}Async(IReadOnlyList<{Input}> inputs, CancellationToken cancellationToken = default);

// Streaming (if applicable)
IAsyncEnumerable<{Token}> {Verb}Async({Input} input, CancellationToken cancellationToken = default);
```

**Domain-Specific Verbs:**

| Package | Primary Verb | Methods |
|---------|--------------|---------|
| Embedder | Embed | `EmbedAsync(text)`, `EmbedAsync(texts)` |
| Reranker | Rerank | `RerankAsync(query, docs)`, `ScoreAsync(query, docs)` |
| Generator | Generate | `GenerateAsync(prompt)`, `GenerateChatAsync(messages)` |
| Transcriber | Transcribe | `TranscribeAsync(audio)`, `TranscribeAsync(stream)` |
| Synthesizer | Synthesize | `SynthesizeAsync(text)` |
| Captioner | Caption | `CaptionAsync(image)` |
| Detector | Detect | `DetectAsync(image)` |

---

## 4. Options Class Pattern

### 4.1 Naming

```csharp
// Pattern: {Domain}Options
public sealed class {Domain}Options
```

### 4.2 Required Properties

All options classes MUST include:

```csharp
public sealed class {Domain}Options
{
    /// <summary>
    /// Directory for caching downloaded models.
    /// Default: ~/.cache/huggingface/hub/
    /// </summary>
    public string? CacheDirectory { get; set; }

    /// <summary>
    /// Execution provider for inference.
    /// Default: ExecutionProvider.Auto
    /// </summary>
    public ExecutionProvider Provider { get; set; } = ExecutionProvider.Auto;
}
```

### 4.3 Common Optional Properties

```csharp
// Sequence/context length (where applicable)
public int? MaxSequenceLength { get; set; }
public int? MaxContextLength { get; set; }

// Performance tuning
public int? ThreadCount { get; set; }
public int BatchSize { get; set; } = 32;

// Download control
public bool DisableAutoDownload { get; set; } = false;
```

---

## 5. Model Aliases

### 5.1 Standard Aliases

All packages MUST support these standard aliases:

| Alias | Purpose |
|-------|---------|
| `default` | Best balance of speed, quality, and size |
| `fast` | Fastest inference, smallest model |
| `quality` | Highest accuracy, larger model |
| `large` | Largest available model |
| `multilingual` | Best multi-language support |

### 5.2 WellKnownModels Registry

```csharp
// Centralized in LocalAI.Generator (or future LocalAI.Common)
public static class WellKnownModels
{
    public static class Embedder { /* ... */ }
    public static class Reranker { /* ... */ }
    public static class Generator { /* ... */ }
    // Future domains...
}
```

---

## 6. Model Information

### 6.1 ModelInfo Record

Each package defines its model information structure:

```csharp
// Pattern: {Domain}ModelInfo
public readonly record struct {Domain}ModelInfo(
    string ModelId,
    string ModelPath,
    int MaxSequenceLength,  // or MaxContextLength
    string ExecutionProvider,
    // Domain-specific properties...
);
```

**Domain-Specific Properties:**

| Package | Additional Properties |
|---------|----------------------|
| Embedder | `Dimensions`, `PoolingMode` |
| Reranker | `Architecture`, `IsMultilingual` |
| Generator | `ChatFormat`, `MaxContextLength` |
| Transcriber | `Language`, `SampleRate` |

---

## 7. Error Handling

### 7.1 Exception Types

```csharp
// Base exception in LocalAI.Core
public class LocalAIException : Exception

// Domain-specific exceptions
public class ModelNotFoundException : LocalAIException
public class ModelLoadException : LocalAIException
public class InferenceException : LocalAIException
```

### 7.2 Validation

- Validate inputs at the earliest point
- Use `ArgumentNullException` for null parameters
- Use `ArgumentException` for invalid values
- Use domain exceptions for model/inference issues

---

## 8. Progress Reporting

All long-running operations MUST support progress reporting:

```csharp
// Shared in LocalAI.Core
public readonly record struct DownloadProgress(
    string FileName,
    long BytesDownloaded,
    long TotalBytes,
    double ProgressPercentage);

// Usage
IProgress<DownloadProgress>? progress
```

---

## 9. Resource Management

### 9.1 Disposable Pattern

All model interfaces MUST implement `IAsyncDisposable`:

```csharp
public interface I{Domain}Model : IAsyncDisposable
{
    // Prefer IAsyncDisposable for async cleanup
}
```

### 9.2 Recommended Usage

```csharp
// Recommended: await using
await using var model = await LocalEmbedder.LoadAsync("default");
var embedding = await model.EmbedAsync("Hello");

// Also valid for sync cleanup
using var model = await LocalEmbedder.LoadAsync("default");
```

---

## 10. Namespace Structure

```
LocalAI.{Domain}/
├── Local{Domain}.cs           # Static factory
├── I{Domain}Model.cs          # Main interface
├── {Domain}Options.cs         # Configuration
├── {Domain}Builder.cs         # Optional builder
├── Models/                    # Domain-specific types
│   ├── {Domain}ModelInfo.cs
│   └── {Output}Result.cs
├── Abstractions/              # Additional interfaces (if needed)
└── Internal/                  # Implementation details
```

---

## 11. Documentation Standards

### 11.1 XML Documentation

All public APIs MUST have XML documentation:

```csharp
/// <summary>
/// Brief description (one line).
/// </summary>
/// <param name="input">Description with examples.</param>
/// <param name="cancellationToken">Cancellation token.</param>
/// <returns>Description of return value.</returns>
/// <exception cref="ArgumentNullException">When input is null.</exception>
/// <example>
/// <code>
/// var result = await model.ProcessAsync("input");
/// </code>
/// </example>
```

### 11.2 Package README

Each package MUST include a README with:
1. Quick start example (2-3 lines)
2. Model aliases table
3. Configuration options
4. Advanced usage examples

---

## 12. Versioning

- All packages share the same version (defined in `Directory.Build.props`)
- Breaking API changes require major version bump
- New features increment minor version
- Bug fixes increment patch version

---

## Appendix A: Current Implementation Status

### Text Packages

| Aspect | Embedder | Reranker | Generator | Translator | Target |
|--------|----------|----------|-----------|------------|--------|
| Entry Point | `LoadAsync` | `LoadAsync` | `LoadAsync` | `LoadAsync` | `LoadAsync` |
| Interface | `IEmbeddingModel` | `IRerankerModel` | `IGeneratorModel` | `ITranslatorModel` | `I{Domain}Model` |
| Options | `EmbedderOptions` | `RerankerOptions` | `GeneratorOptions` | `TranslatorOptions` | `{Domain}Options` |
| Builder | - | - | `TextGeneratorBuilder` | - | Optional |
| `WarmupAsync` | ✅ | ✅ | ✅ | ✅ | Required |
| `GetModelInfo` | ✅ | ✅ | ✅ | ✅ | Required |
| `IAsyncDisposable` | ✅ | ✅ | ✅ | ✅ | Required |
| Aliases | ✅ | ✅ | ✅ | ✅ | Required |

### Vision Packages

| Aspect | Captioner | Ocr | Detector | Segmenter | Target |
|--------|-----------|-----|----------|-----------|--------|
| Entry Point | `LoadAsync` | `LoadAsync` | `LoadAsync` | `LoadAsync` | `LoadAsync` |
| Interface | `ICaptionerModel` | `IOcrModel` | `IDetectorModel` | `ISegmenterModel` | `I{Domain}Model` |
| Options | `CaptionerOptions` | `OcrOptions` | `DetectorOptions` | `SegmenterOptions` | `{Domain}Options` |
| `WarmupAsync` | ✅ | ✅ | ✅ | ✅ | Required |
| `GetModelInfo` | ✅ | ✅ | ✅ | ✅ | Required |
| `IAsyncDisposable` | ✅ | ✅ | ✅ | ✅ | Required |
| Aliases | ✅ | ✅ | ✅ | ✅ | Required |

### Remaining Work

All current packages are fully compliant with API design guidelines.

---

## Appendix B: Future Package Template

```csharp
// LocalAI.Transcriber example

public static class LocalTranscriber
{
    public static Task<ITranscriberModel> LoadAsync(
        string modelIdOrPath = "default",
        TranscriberOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default);
}

public interface ITranscriberModel : IAsyncDisposable
{
    string ModelId { get; }

    Task<TranscriptionResult> TranscribeAsync(
        Stream audioStream,
        TranscribeOptions? options = null,
        CancellationToken cancellationToken = default);

    IAsyncEnumerable<TranscriptionSegment> TranscribeStreamingAsync(
        Stream audioStream,
        TranscribeOptions? options = null,
        CancellationToken cancellationToken = default);

    Task WarmupAsync(CancellationToken cancellationToken = default);
    TranscriberModelInfo? GetModelInfo();
}

public sealed class TranscriberOptions
{
    public string? CacheDirectory { get; set; }
    public ExecutionProvider Provider { get; set; } = ExecutionProvider.Auto;
    public string? Language { get; set; } // Domain-specific
    public bool EnableTimestamps { get; set; } = true; // Domain-specific
}
```

---

*Last Updated: 2025-12*
