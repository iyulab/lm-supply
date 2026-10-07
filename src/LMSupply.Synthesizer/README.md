# LMSupply.Synthesizer

Local text-to-speech synthesis using VITS/Piper models.

> **Known issue — the output is not yet intelligible speech.** The synthesizer has no text-to-phoneme step: it maps letters to fixed ids instead of the phoneme ids the Piper voices were trained on, so what comes out is voice-like noise (a Whisper transcript of "The weather is beautiful today." read back "tube of warrior practitioner"), and text in non-Latin scripts comes out as near-silence.

## Features

- **Zero-config**: Models download automatically from HuggingFace
- **GPU Acceleration**: CUDA, DirectML (Windows), CoreML (macOS)
- **Licensing stated per voice**: Piper's code is MIT; each voice's own license is in `SynthesizerModelInfo.License` (the default voice is public domain)
- **Multiple Languages**: English, Korean, Japanese, Chinese, and more

## Quick Start

```csharp
using LMSupply.Synthesizer;

#pragma warning disable LMSUPPLY001 // the output is not yet intelligible speech — see the repository README's known issue
// Load the default model
await using var synthesizer = await LocalSynthesizer.LoadAsync("default");

// Synthesize speech
var result = await synthesizer.SynthesizeAsync("Hello, welcome to LMSupply!");

Console.WriteLine($"Duration: {result.DurationSeconds:F2}s");
Console.WriteLine($"Real-time factor: {result.RealTimeFactor:F1}x");

// Save as WAV file
await synthesizer.SynthesizeToFileAsync("Hello world!", "output.wav");
```

## Available Models

| Alias | Voice | Language | Description |
|-------|-------|----------|-------------|
| `default` | LJSpeech | en-US | Female (public domain) |
| `lessac` | Lessac | en-US | High-quality female |
| `fast` | Ryan | en-US | Fast male voice |
| `quality` | Amy | en-US | High-quality female |
| `british` | Semaine | en-GB | British female |
| `korean` | KSS | ko-KR | Korean female |
| `chinese` | Huayan | zh-CN | Mandarin female |

## GPU Acceleration

Do not add ONNX Runtime packages (`Microsoft.ML.OnnxRuntime*`): LMSupply provisions the runtime itself, and a
second copy conflicts with it. `ExecutionProvider.Auto` uses CUDA when the CUDA 12 runtime and cuDNN 9 are
installed on the machine, CoreML on macOS, and the CPU otherwise. On Windows with an AMD or Intel GPU, ONNX
sessions run on the CPU (DirectML was removed in 0.67.0). See
[GPU acceleration](https://github.com/iyulab/lm-supply#gpu-acceleration).
