# LMSupply.Synthesizer

Local text-to-speech synthesis using VITS/Piper models.

## Features

- **Zero-config**: Models download automatically from HuggingFace
- **English text out of the box**: a built-in English phonemizer (pronunciation dictionary + letter-to-sound rules, in
  the package, offline) writes each English voice's own phonemes
- **The voice's own contract**: phoneme ids, inference defaults, sample rate and speaker count are read from the voice's config
- **Phoneme input and a phonemizer seam**: pass phonemes (`SynthesizePhonemesAsync`), or plug in the text-to-phoneme step (`IPhonemizer`)
- **GPU Acceleration**: CUDA (NVIDIA), CoreML (macOS)
- **Licensing stated per voice**: Piper's code is MIT; each voice's own license is in `SynthesizerModelInfo.License` (the default voice is public domain)
- **Multiple Languages**: English (text and phonemes), Korean and Chinese (phonemes, or text through your phonemizer)

## Quick Start

```csharp
using LMSupply.Synthesizer;

// Load the default model
await using var synthesizer = await LocalSynthesizer.LoadAsync("default");

var result = await synthesizer.SynthesizeAsync("The weather is beautiful today.");

Console.WriteLine($"Duration: {result.DurationSeconds:F2}s");
Console.WriteLine($"Real-time factor: {result.RealTimeFactor:F1}x");

// Save as WAV file
await File.WriteAllBytesAsync("output.wav", result.ToWavBytes());
```

## Text

Text for the English voices is turned into phonemes by the package: numbers, a few common abbreviations and the
punctuation that makes pauses are read, each word is pronounced from the CMU Pronouncing Dictionary or, if it is not
there, by letter-to-sound rules, and the result is written in the voice's own accent (American for `en-us`, British-style
for the default voice's `en` and for `en-gb-x-rp`). A word with two readings gets its most common one. See the
[synthesizer guide](https://github.com/iyulab/lm-supply/blob/main/docs/synthesizer.md#english-text) for what is and is
not handled.

For any other voice, set an `IPhonemizer` in `SynthesizerOptions.Phonemizer` (it replaces the built-in one for every
voice); the model passes it the phonemes its voice declares (`ISynthesizerModel.PhonemeScheme`: the type, the phonemizer
voice and the language).

```csharp
using LMSupply.Synthesizer;

var options = new SynthesizerOptions { Phonemizer = new MyPhonemizer() };
await using var synthesizer = await LocalSynthesizer.LoadAsync("korean", options);

var text = File.ReadAllText("input.txt"); // Korean text
await synthesizer.SynthesizeToFileAsync(text, "output.wav");

sealed class MyPhonemizer : IPhonemizer
{
    public ValueTask<IReadOnlyList<string>> PhonemizeAsync(
        string text, PhonemeScheme scheme, CancellationToken cancellationToken = default)
    {
        // One string per sentence, in the voice's scheme (here "espeak" / "ko"); every code point is a phoneme,
        // spaces and clause punctuation included.
        IReadOnlyList<string> sentences = [/* your text-to-phoneme step's output */];
        return new(sentences);
    }
}
```

Without a phonemizer, text for a voice that is neither English nor of the `text` phoneme type (the Korean and Chinese
voices) throws `NotSupportedException`; it is never spoken as noise.

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
second copy conflicts with it. `ExecutionProvider.Auto` uses CUDA when the driver supports CUDA 13 and the CUDA 13
runtime and cuDNN 9 are installed on the machine (ONNX Runtime 1.30 is built for CUDA 13), CoreML on macOS, and the CPU otherwise. On Windows with an AMD or Intel GPU, ONNX
sessions run on the CPU (DirectML was removed in 0.67.0). See
[GPU acceleration](https://github.com/iyulab/lm-supply#gpu-acceleration).
