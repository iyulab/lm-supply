# LMSupply.Translator

A lightweight, zero-configuration neural machine translation library for .NET with automatic GPU acceleration.

## Installation

```bash
dotnet add package LMSupply.Translator
```

For GPU acceleration:

```bash
# NVIDIA CUDA
dotnet add package Microsoft.ML.OnnxRuntime.Gpu

# Windows DirectML
dotnet add package Microsoft.ML.OnnxRuntime.DirectML

# macOS CoreML
dotnet add package Microsoft.ML.OnnxRuntime.CoreML
```

## Basic Usage

```csharp
using LMSupply.Translator;

// Load a translation model (Korean to English)
await using var translator = await LocalTranslator.LoadAsync("ko-en");

// Translate text
var result = await translator.TranslateAsync("안녕하세요, 반갑습니다.");
Console.WriteLine(result.TranslatedText);
// Output: "Hello, nice to meet you."

Console.WriteLine($"Source: {result.SourceLanguage} → Target: {result.TargetLanguage}");
// Output: "Source: ko → Target: en"
```

## Available Models

| Alias | Direction | Model | Size | BLEU | Description |
|-------|-----------|-------|------|------|-------------|
| `default` | Ko → En | OPUS-MT | ~300MB | 35.5 | Korean to English (default) |
| `ko-en` | Ko → En | OPUS-MT | ~300MB | 35.5 | Korean to English |
| `en-ko` | En → Ko | OPUS-MT | ~300MB | 28.0 | English to Korean |
| `ja-en` | Ja → En | OPUS-MT | ~300MB | 32.0 | Japanese to English |
| `zh-en` | Zh → En | OPUS-MT | ~300MB | 30.5 | Chinese to English |

All models use Apache-2.0 license and are based on MarianMT architecture.

## Advanced Usage

### Custom Options

```csharp
var options = new TranslatorOptions
{
    MaxLength = 256,                       // Maximum output length
    BeamWidth = 4,                         // Beam search width (higher = better quality)
    LengthPenalty = 1.0f,                  // Favor longer (>1) or shorter (<1) translations
    RepetitionPenalty = 1.2f,              // Penalize repeated tokens
    Provider = ExecutionProvider.DirectML, // Force specific GPU provider
    CacheDirectory = "/custom/cache"       // Custom model cache directory
};

var translator = await LocalTranslator.LoadAsync("ko-en", options);
```

### Greedy vs Beam Search Decoding

```csharp
// Faster but potentially lower quality
var greedyOptions = new TranslatorOptions
{
    UseGreedyDecoding = true  // Skip beam search for faster inference
};

var fastTranslator = await LocalTranslator.LoadAsync("ko-en", greedyOptions);

// Higher quality with beam search (default)
var qualityOptions = new TranslatorOptions
{
    BeamWidth = 5,           // More beams = better quality
    LengthPenalty = 1.2f     // Slightly favor longer translations
};

var qualityTranslator = await LocalTranslator.LoadAsync("ko-en", qualityOptions);
```

### Batch Translation

```csharp
var texts = new[]
{
    "안녕하세요",
    "감사합니다",
    "좋은 하루 되세요"
};

var results = await translator.TranslateBatchAsync(texts);

foreach (var result in results)
{
    Console.WriteLine($"{result.SourceText} → {result.TranslatedText}");
}
// Output:
// 안녕하세요 → Hello
// 감사합니다 → Thank you
// 좋은 하루 되세요 → Have a nice day
```

### Translation with Metadata

```csharp
var result = await translator.TranslateAsync("오늘 날씨가 좋습니다.");

Console.WriteLine($"Source: {result.SourceText}");
Console.WriteLine($"Translation: {result.TranslatedText}");
Console.WriteLine($"Direction: {result.SourceLanguage} → {result.TargetLanguage}");

if (result.Confidence.HasValue)
    Console.WriteLine($"Confidence: {result.Confidence:P1}");

if (result.InferenceTimeMs.HasValue)
    Console.WriteLine($"Time: {result.InferenceTimeMs:F0}ms");
```

### Checking Language Support

```csharp
// Get all available model aliases
var models = LocalTranslator.GetAvailableModels();
Console.WriteLine(string.Join(", ", models));
// Output: default, ko-en, en-ko, ja-en, zh-en

// Get detailed model information
var allModels = LocalTranslator.GetAllModels();
foreach (var model in allModels)
{
    Console.WriteLine($"{model.Alias}: {model.SourceLanguage} → {model.TargetLanguage}");
}
```

### Multiple Translation Directions

```csharp
// Load translators for different directions
await using var koToEn = await LocalTranslator.LoadAsync("ko-en");
await using var enToKo = await LocalTranslator.LoadAsync("en-ko");

// Korean to English
var english = await koToEn.TranslateAsync("안녕하세요");
Console.WriteLine(english.TranslatedText); // "Hello"

// English to Korean
var korean = await enToKo.TranslateAsync("Hello");
Console.WriteLine(korean.TranslatedText); // "안녕하세요"
```

## GPU Acceleration

GPU acceleration is automatic when available. Priority order:
1. CUDA (NVIDIA GPUs)
2. DirectML (Windows - AMD, Intel, NVIDIA)
3. CoreML (macOS)
4. CPU (fallback)

Force a specific provider:

```csharp
var options = new TranslatorOptions
{
    Provider = ExecutionProvider.Cuda
};
```

## Model Caching

Models are cached following HuggingFace Hub conventions:
- Default: `~/.cache/huggingface/hub`
- Override via: `HF_HUB_CACHE`, `HF_HOME`, or `XDG_CACHE_HOME` environment variables
- Or set `TranslatorOptions.CacheDirectory`

## Architecture Notes

The translator uses encoder-decoder MarianMT architecture with:
- **Encoder**: Processes source language input
- **Decoder**: Generates target language output autoregressively
- **Beam Search**: Explores multiple hypotheses for better translations
- **SentencePiece Tokenizer**: Handles subword tokenization for all languages

## Performance Tips

1. **Use greedy decoding for real-time applications**: Set `UseGreedyDecoding = true`
2. **Batch multiple texts**: Use `TranslateBatchAsync` for throughput
3. **Adjust beam width**: Lower `BeamWidth` (1-2) for speed, higher (4-6) for quality
4. **Enable GPU**: Ensure ONNX Runtime GPU package is installed
5. **Warmup the model**: Call `WarmupAsync()` before time-critical translations
