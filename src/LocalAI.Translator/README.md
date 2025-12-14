# LocalAI.Translator

Local neural machine translation for .NET with automatic model downloading.

## Features

- **Zero-config**: Models download automatically from HuggingFace
- **GPU Acceleration**: CUDA, DirectML (Windows), CoreML (macOS)
- **Apache-2.0 Licensed**: OPUS-MT models for commercial use
- **Asian Languages**: Korean, Japanese, Chinese to/from English

## Quick Start

```csharp
using LocalAI.Translator;

// Load a translation model (Korean to English)
await using var translator = await LocalTranslator.LoadAsync("ko-en");

// Translate text
var result = await translator.TranslateAsync("안녕하세요, 반갑습니다.");

Console.WriteLine(result.TranslatedText);
// Output: "Hello, nice to meet you."

Console.WriteLine($"{result.SourceLanguage} → {result.TargetLanguage}");
```

## Available Models

| Alias | Direction | Model | BLEU | Description |
|-------|-----------|-------|------|-------------|
| `default` | Ko → En | OPUS-MT | 35.5 | Default |
| `ko-en` | Ko → En | OPUS-MT | 35.5 | Korean to English |
| `en-ko` | En → Ko | OPUS-MT | 28.0 | English to Korean |
| `ja-en` | Ja → En | OPUS-MT | 32.0 | Japanese to English |
| `zh-en` | Zh → En | OPUS-MT | 30.5 | Chinese to English |

## GPU Acceleration

```bash
# NVIDIA GPU
dotnet add package Microsoft.ML.OnnxRuntime.Gpu

# Windows (AMD/Intel/NVIDIA)
dotnet add package Microsoft.ML.OnnxRuntime.DirectML
```
