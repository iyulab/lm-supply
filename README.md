# LocalAI

[![CI](https://github.com/iyulab/local-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/iyulab/local-ai/actions/workflows/ci.yml)
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
| [LocalAI.Captioner](docs/captioner.md) | Image → Text captioning | [![NuGet](https://img.shields.io/nuget/v/LocalAI.Captioner.svg)](https://www.nuget.org/packages/LocalAI.Captioner) |
| [LocalAI.Ocr](docs/ocr.md) | Document OCR | [![NuGet](https://img.shields.io/nuget/v/LocalAI.Ocr.svg)](https://www.nuget.org/packages/LocalAI.Ocr) |
| [LocalAI.Detector](docs/detector.md) | Object detection | [![NuGet](https://img.shields.io/nuget/v/LocalAI.Detector.svg)](https://www.nuget.org/packages/LocalAI.Detector) |
| [LocalAI.Segmenter](docs/segmenter.md) | Image segmentation | [![NuGet](https://img.shields.io/nuget/v/LocalAI.Segmenter.svg)](https://www.nuget.org/packages/LocalAI.Segmenter) |
| [LocalAI.Translator](docs/translator.md) | Neural machine translation | [![NuGet](https://img.shields.io/nuget/v/LocalAI.Translator.svg)](https://www.nuget.org/packages/LocalAI.Translator) |
| [LocalAI.Transcriber](docs/transcriber.md) | Speech → Text (Whisper) | [![NuGet](https://img.shields.io/nuget/v/LocalAI.Transcriber.svg)](https://www.nuget.org/packages/LocalAI.Transcriber) |
| [LocalAI.Synthesizer](docs/synthesizer.md) | Text → Speech (Piper) | [![NuGet](https://img.shields.io/nuget/v/LocalAI.Synthesizer.svg)](https://www.nuget.org/packages/LocalAI.Synthesizer) |

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

### Translation

```csharp
using LocalAI.Translator;

await using var translator = await LocalTranslator.LoadAsync("ko-en");

// Translate Korean to English
string english = await translator.TranslateAsync("안녕하세요, 세계!");
Console.WriteLine(english); // "Hello, world!"

// Batch translation
string[] translations = await translator.TranslateBatchAsync(new[]
{
    "첫 번째 문장입니다.",
    "두 번째 문장입니다."
});
```

### Speech Recognition (Transcriber)

```csharp
using LocalAI.Transcriber;

await using var transcriber = await LocalTranscriber.LoadAsync("default");

// Transcribe audio file
var result = await transcriber.TranscribeAsync("audio.wav");
Console.WriteLine(result.Text);
Console.WriteLine($"Language: {result.Language}");

// Streaming transcription
await foreach (var segment in transcriber.TranscribeStreamingAsync("audio.wav"))
{
    Console.WriteLine($"[{segment.Start:F2}s] {segment.Text}");
}
```

### Text-to-Speech (Synthesizer)

```csharp
using LocalAI.Synthesizer;

await using var synthesizer = await LocalSynthesizer.LoadAsync("default");

// Synthesize and save to file
await synthesizer.SynthesizeToFileAsync("Hello, world!", "output.wav");

// Get audio samples
var result = await synthesizer.SynthesizeAsync("Hello!");
Console.WriteLine($"Duration: {result.DurationSeconds:F2}s");
Console.WriteLine($"Real-time factor: {result.RealTimeFactor:F1}x");
```

---

## Available Models

*Updated: 2025-01 based on MTEB leaderboard and community benchmarks*

### Embedder

| Alias | Model | Dims | Params | Context | Best For |
|-------|-------|------|--------|---------|----------|
| `default` | bge-small-en-v1.5 | 384 | 33M | 512 | Balanced speed/quality |
| `fast` | all-MiniLM-L6-v2 | 384 | 22M | 256 | Ultra-low latency |
| `quality` | bge-base-en-v1.5 | 768 | 110M | 512 | Higher accuracy |
| `large` | nomic-embed-text-v1.5 | 768 | 137M | 8192 | Long context RAG |
| `multilingual` | multilingual-e5-base | 768 | 278M | 512 | 100+ languages |

### Reranker

| Alias | Model | Params | Context | Best For |
|-------|-------|--------|---------|----------|
| `default` | ms-marco-MiniLM-L-6-v2 | 22M | 512 | Balanced speed/quality |
| `fast` | ms-marco-TinyBERT-L-2-v2 | 4.4M | 512 | Ultra-low latency |
| `quality` | bge-reranker-base | 278M | 512 | Higher accuracy |
| `large` | bge-reranker-large | 560M | 512 | Best accuracy |
| `multilingual` | bge-reranker-v2-m3 | 568M | 8192 | Long docs, 100+ languages |

### Generator

| Alias | Model | Params | Context | License | Best For |
|-------|-------|--------|---------|---------|----------|
| `default` | Phi-4-mini-instruct | 3.8B | 16K | MIT | Balanced reasoning |
| `fast` | Llama-3.2-1B-Instruct | 1B | 8K | Llama 3.2 | Ultra-fast inference |
| `quality` | phi-4 | 14B | 16K | MIT | Best reasoning |
| `medium` | Phi-3.5-mini-instruct | 3.8B | 128K | MIT | Long context |
| `multilingual` | gemma-2-2b-it | 2B | 8K | Gemma ToU | Multi-language |

### Translator

| Alias | Direction | Model | Best For |
|-------|-----------|-------|----------|
| `ko-en` | Korean → English | OPUS-MT | Korean translation |
| `en-ko` | English → Korean | OPUS-MT | Korean translation |
| `ja-en` | Japanese → English | OPUS-MT | Japanese translation |
| `zh-en` | Chinese → English | OPUS-MT | Chinese translation |
| `multilingual` | Many → English | mBART/M2M100 | 100+ languages |

### Transcriber (Whisper)

| Alias | Model | Params | Size | WER | Best For |
|-------|-------|--------|------|-----|----------|
| `fast` | Whisper Tiny | 39M | ~150MB | 7.6% | Ultra-fast transcription |
| `default` | Whisper Base | 74M | ~290MB | 5.0% | Balanced speed/quality |
| `quality` | Whisper Small | 244M | ~970MB | 3.4% | Higher accuracy |
| `large` | Whisper Large V3 | 1.5B | ~6GB | 2.5% | Best accuracy |
| `english` | Whisper Base.en | 74M | ~290MB | 4.3% | English-optimized |

### Synthesizer (Piper TTS)

| Alias | Voice | Language | Sample Rate | Best For |
|-------|-------|----------|-------------|----------|
| `default` | Lessac | en-US | 22050 Hz | Balanced quality |
| `fast` | Ryan | en-US | 16000 Hz | Ultra-fast synthesis |
| `quality` | Amy | en-US | 22050 Hz | High quality |
| `british` | Semaine | en-GB | 22050 Hz | British English |
| `korean` | KSS | ko-KR | 22050 Hz | Korean |
| `japanese` | JSUT | ja-JP | 22050 Hz | Japanese |
| `chinese` | Huayan | zh-CN | 22050 Hz | Mandarin Chinese |

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

### Text & Language
- [Embedder Guide](docs/embedder.md) - Text → Vector embeddings
- [Reranker Guide](docs/reranker.md) - Semantic reranking
- [Generator Guide](docs/generator.md) - Text generation & chat
- [Translator Guide](docs/translator.md) - Neural machine translation

### Vision
- [Captioner Guide](docs/captioner.md) - Image → Text captioning
- [OCR Guide](docs/ocr.md) - Document text recognition
- [Detector Guide](docs/detector.md) - Object detection
- [Segmenter Guide](docs/segmenter.md) - Image segmentation

### Audio
- [Transcriber Guide](docs/transcriber.md) - Speech → Text (Whisper)
- [Synthesizer Guide](docs/synthesizer.md) - Text → Speech (Piper)

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
