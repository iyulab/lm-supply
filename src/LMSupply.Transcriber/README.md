# LMSupply.Transcriber

Local speech-to-text transcription using Whisper models.

## Features

- **Zero-config**: Models download automatically from HuggingFace
- **GPU Acceleration**: CUDA, DirectML (Windows), CoreML (macOS)
- **MIT Licensed**: OpenAI Whisper models
- **99+ Languages**: Auto-detection and multilingual support

## Quick Start

```csharp
using LMSupply.Transcriber;

// Load the default model
await using var transcriber = await LocalTranscriber.LoadAsync("default");

// Transcribe audio
var result = await transcriber.TranscribeAsync("audio.wav");

Console.WriteLine(result.Text);
Console.WriteLine($"Language: {result.Language}");
Console.WriteLine($"Duration: {result.DurationSeconds}s");
```

## Available Models

| Alias | Model | Size | WER | Description |
|-------|-------|------|-----|-------------|
| `fast` | Whisper Tiny | ~150MB | 7.6% | Ultra-fast |
| `default` | Whisper Base | ~290MB | 5.0% | Balanced |
| `quality` | Whisper Small | ~970MB | 3.4% | Higher accuracy |
| `medium` | Whisper Medium | ~3GB | 2.9% | High quality |
| `large` | Whisper Large V3 | ~6GB | 2.5% | Highest accuracy |
| `english` | Whisper Base.en | ~290MB | 4.3% | English optimized |
| `parakeet-tdt` | Parakeet TDT 0.6B v3 (int8) | ~670MB | — | NVIDIA transducer, 25 European languages, no 30 s window, no translation. Opt-in only (never chosen by `auto`); `Language` is `"und"` unless you pass a hint — the model has no language-id output |

## GPU Acceleration

```bash
# NVIDIA GPU
dotnet add package Microsoft.ML.OnnxRuntime.Gpu

# Windows (AMD/Intel/NVIDIA)
dotnet add package Microsoft.ML.OnnxRuntime.DirectML
```
