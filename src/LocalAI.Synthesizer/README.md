# LocalAI.Synthesizer

Local text-to-speech synthesis using VITS/Piper models.

## Features

- **Zero-config**: Models download automatically from HuggingFace
- **GPU Acceleration**: CUDA, DirectML (Windows), CoreML (macOS)
- **MIT Licensed**: Piper VITS voices
- **Multiple Languages**: English, Korean, Japanese, Chinese, and more

## Quick Start

```csharp
using LocalAI.Synthesizer;

// Load the default model
await using var synthesizer = await LocalSynthesizer.LoadAsync("default");

// Synthesize speech
var result = await synthesizer.SynthesizeAsync("Hello, welcome to LocalAI!");

Console.WriteLine($"Duration: {result.DurationSeconds:F2}s");
Console.WriteLine($"Real-time factor: {result.RealTimeFactor:F1}x");

// Save as WAV file
await synthesizer.SynthesizeToFileAsync("Hello world!", "output.wav");
```

## Available Models

| Alias | Voice | Language | Description |
|-------|-------|----------|-------------|
| `default` | Lessac | en-US | High-quality female |
| `fast` | Ryan | en-US | Fast male voice |
| `quality` | Amy | en-US | High-quality female |
| `british` | Semaine | en-GB | British female |
| `korean` | KSS | ko-KR | Korean female |
| `japanese` | JSUT | ja-JP | Japanese female |
| `chinese` | Huayan | zh-CN | Mandarin female |

## GPU Acceleration

```bash
# NVIDIA GPU
dotnet add package Microsoft.ML.OnnxRuntime.Gpu

# Windows (AMD/Intel/NVIDIA)
dotnet add package Microsoft.ML.OnnxRuntime.DirectML
```
