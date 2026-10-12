# LMSupply.Synthesizer

Local text-to-speech synthesis using VITS/Piper models.

> **Known issue — text needs a phonemizer the package does not ship yet.** A voice is run by the contract its own config declares, so phonemes (`SynthesizePhonemesAsync`), or text through an `IPhonemizer` you set in `SynthesizerOptions.Phonemizer`, come out as real speech. But every voice in the catalog was trained on eSpeak NG phonemes, and the package has no text-to-phoneme step for them, so `SynthesizeAsync(text)` with one of those voices throws `NotSupportedException` unless you supply a phonemizer — it does not return audio.

## Features

- **Zero-config**: Models download automatically from HuggingFace
- **The voice's own contract**: phoneme ids, inference defaults, sample rate and speaker count are read from the voice's config
- **Phoneme input and a phonemizer seam**: pass phonemes (`SynthesizePhonemesAsync`), or plug in the text-to-phoneme step (`IPhonemizer`)
- **GPU Acceleration**: CUDA, DirectML (Windows), CoreML (macOS)
- **Licensing stated per voice**: Piper's code is MIT; each voice's own license is in `SynthesizerModelInfo.License` (the default voice is public domain)
- **Multiple Languages**: English, Korean, Chinese

## Quick Start

```csharp
using LMSupply.Synthesizer;

#pragma warning disable LMSUPPLY001 // text needs a phonemizer the package does not ship yet — see the known issue above
// Load the default model
await using var synthesizer = await LocalSynthesizer.LoadAsync("default");

// Synthesize speech from phonemes of the voice's scheme (eSpeak NG's IPA for this voice), one string per sentence
var result = await synthesizer.SynthesizePhonemesAsync(["ðə wˈɛðəɹ ɪz bjˈuːtifəl tədˈeɪ."]);

Console.WriteLine($"Duration: {result.DurationSeconds:F2}s");
Console.WriteLine($"Real-time factor: {result.RealTimeFactor:F1}x");

// Save as WAV file
await File.WriteAllBytesAsync("output.wav", result.ToWavBytes());
```

## Text

Text is turned into phonemes by the `IPhonemizer` set in `SynthesizerOptions.Phonemizer`; the model passes it the
phonemes its voice declares (`ISynthesizerModel.PhonemeScheme`: the type, the phonemizer voice and the language).

```csharp
using LMSupply.Synthesizer;

#pragma warning disable LMSUPPLY001
var options = new SynthesizerOptions { Phonemizer = new MyPhonemizer() };
await using var synthesizer = await LocalSynthesizer.LoadAsync("default", options);

await synthesizer.SynthesizeToFileAsync("Hello, world!", "output.wav");

sealed class MyPhonemizer : IPhonemizer
{
    public ValueTask<IReadOnlyList<string>> PhonemizeAsync(
        string text, PhonemeScheme scheme, CancellationToken cancellationToken = default)
    {
        // One string per sentence; every code point is a phoneme, spaces and clause punctuation included.
        IReadOnlyList<string> sentences = ["həlˈəʊ, wˈɜːld!"];
        return new(sentences);
    }
}
```

Without a phonemizer, text works only for a voice whose phoneme type is `text` (its phonemes are its characters). For
any other voice it throws `NotSupportedException`.

## Available Models

| Alias | Voice | Language | Description |
|-------|-------|----------|-------------|
| `default` | LJSpeech | en-US | Female (public domain) |
| `lessac` | Lessac | en-US | High-quality female |
| `fast` | Ryan | en-US | Fast male voice |
| `quality` | Amy | en-US | High-quality female |
| `british` | Semaine | en-GB | British, four speakers |
| `korean` | KSS | ko-KR | Korean female |
| `chinese` | Huayan | zh-CN | Mandarin female |

## GPU Acceleration

Do not add ONNX Runtime packages (`Microsoft.ML.OnnxRuntime*`): LMSupply provisions the runtime itself, and a
second copy conflicts with it. `ExecutionProvider.Auto` uses CUDA when the CUDA 12 runtime and cuDNN 9 are
installed on the machine, CoreML on macOS, and the CPU otherwise. On Windows with an AMD or Intel GPU, ONNX
sessions run on the CPU (DirectML was removed in 0.67.0). See
[GPU acceleration](https://github.com/iyulab/lm-supply#gpu-acceleration).
