# LMSupply.Translator

Local neural machine translation for .NET with automatic model downloading.

## Features

- **Zero-config**: Models download automatically from HuggingFace
- **GPU Acceleration**: CUDA, DirectML (Windows), CoreML (macOS)
- **Apache-2.0 Licensed**: OPUS-MT models for commercial use
- **Asian Languages**: Korean, Japanese, Chinese to/from English

## Quick Start

```csharp
using LMSupply.Translator;

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
| `ja-en` | Ja → En | OPUS-MT | 32.0 | Japanese to English |
| `zh-en` | Zh → En | OPUS-MT | 30.5 | Chinese to English |

## GPU Acceleration

Do not add ONNX Runtime packages (`Microsoft.ML.OnnxRuntime*`): LMSupply provisions the runtime itself, and a
second copy conflicts with it. `ExecutionProvider.Auto` uses CUDA when the CUDA 12 runtime and cuDNN 9 are
installed on the machine, CoreML on macOS, and the CPU otherwise. On Windows with an AMD or Intel GPU, ONNX
sessions run on the CPU (DirectML was removed in 0.67.0). See
[GPU acceleration](https://github.com/iyulab/lm-supply#gpu-acceleration).
