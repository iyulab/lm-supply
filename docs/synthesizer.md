# LMSupply.Synthesizer

Text-to-speech synthesis using VITS/Piper models with ONNX Runtime.

> **Known issue — text needs a phonemizer the package does not ship yet.** A voice is run by the contract its own config declares, so phonemes (`SynthesizePhonemesAsync`), or text through an `IPhonemizer` you set in `SynthesizerOptions.Phonemizer`, come out as real speech: a Whisper transcript of "The weather is beautiful today." spoken from its phonemes reads back word for word. But every voice in the catalog was trained on eSpeak NG phonemes, and the package has no text-to-phoneme step for them, so `SynthesizeAsync(text)` with one of those voices throws `NotSupportedException` unless you supply a phonemizer — it does not return audio. Everything else (loading, voices, WAV output, streaming) works. Every use of `LocalSynthesizer` / `ISynthesizerModel` reports `LMSUPPLY001` (`[Experimental]`, an error by default) until a phonemizer ships with the package; suppress that one id to use the package on those terms.

## Quick Start

```csharp
using LMSupply.Synthesizer;

// Load the default model (English US LJSpeech, public domain)
var synthesizer = await LocalSynthesizer.LoadAsync("default");

// Synthesize speech from phonemes of the voice's scheme (eSpeak NG's IPA for this voice): one string per sentence
var result = await synthesizer.SynthesizePhonemesAsync(["ðə wˈɛðəɹ ɪz bjˈuːtifəl tədˈeɪ."]);
Console.WriteLine($"Duration: {result.DurationSeconds:F2}s");
Console.WriteLine($"Real-time factor: {result.RealTimeFactor:F1}x");

// Save as WAV file
await File.WriteAllBytesAsync("output.wav", result.ToWavBytes());
```

To synthesize from text, set a phonemizer — see [Phonemes and phonemizers](#phonemes-and-phonemizers).

## Installation

```bash
dotnet add package LMSupply.Synthesizer
```

## Features

- **VITS Models**: High-quality neural TTS using Piper voices
- **The voice's own contract**: phoneme ids, inference defaults, sample rate and speaker count are read from the voice's config
- **Phoneme input and a phonemizer seam**: pass phonemes, or plug in the text-to-phoneme step a voice needs
- **Multiple Languages**: English, Korean, Chinese
- **Fast Inference**: Optimized ONNX models for real-time synthesis
- **Streaming**: Generate audio sentence by sentence
- **Multiple Formats**: WAV, raw PCM16, and float32 output
- **GPU Acceleration**: CUDA and CoreML support

## Available Models

| Alias | Voice | Language | Sample Rate | Size | Phonemes | Description |
|-------|-------|----------|-------------|------|----------|-------------|
| `default` | LJSpeech | en-US | 22050 Hz | ~64MB | `espeak` / `en` | US English female (public domain) |
| `lessac` | Lessac | en-US | 22050 Hz | ~64MB | `espeak` / `en-us` | High-quality US English female |
| `fast` | Ryan | en-US | 16000 Hz | ~16MB | `espeak` / `en-us` | Fast US English male |
| `quality` | Amy | en-US | 22050 Hz | ~64MB | `espeak` / `en-us` | High-quality US English female |
| `british` | Semaine | en-GB | 22050 Hz | ~64MB | `espeak` / `en-gb-x-rp` | British English, four speakers |
| `korean` | KSS (medium) | ko-KR | 22050 Hz | ~63MB | `espeak` / `ko` | Korean female |
| `chinese` | Huayan | zh-CN | 22050 Hz | ~64MB | `espeak` / `cmn` | Mandarin Chinese female |

The Phonemes column is what each voice's config declares (`phoneme_type` / `espeak.voice`); a loaded model reports it
as `PhonemeScheme`.

## Phonemes and phonemizers

A Piper voice does not read letters. Its config (the `.onnx.json` next to the model) declares the phonemes it was
trained on and the id of each one, and the synthesizer encodes exactly that: the start symbol, a pad, every phoneme
followed by a pad, the end symbol. `ISynthesizerModel.PhonemeScheme` reports what the voice declares:

```csharp
PhonemeScheme scheme = synthesizer.PhonemeScheme;
Console.WriteLine(scheme.Type);      // "espeak": IPA as eSpeak NG writes it; or "text": the characters of the text
Console.WriteLine(scheme.Voice);     // the phonemizer voice the config names, e.g. "en", "en-us", "en-gb-x-rp"
Console.WriteLine(scheme.Language);  // e.g. "en-US"
```

### Phonemes as input

Phonemes are passed one string per sentence. In a string every Unicode code point is one phoneme — a space is a
phoneme, and so is the punctuation that ends a clause or the sentence (`, ; : . ? !`). Each sentence is synthesized on
its own.

```csharp
var result = await synthesizer.SynthesizePhonemesAsync(
[
    "ðə wˈɛðəɹ ɪz bjˈuːtifəl tədˈeɪ.",
    "ðə kwˈɪk bɹˈaʊn fˈɒks dʒˈʌmps ˌəʊvə ðə lˈeɪzi dˈɒɡ.",
]);

// Phonemes the voice has no id for are skipped, not guessed at
if (result.UnknownPhonemes.Length > 0)
    Console.WriteLine($"Skipped: {result.UnknownPhonemes}");
```

### Text through a phonemizer

`IPhonemizer` is the text-to-phoneme step. It is given the text and the voice's `PhonemeScheme`, and returns the
phonemes in the same form `SynthesizePhonemesAsync` takes. Set it when loading the model:

```csharp
var options = new SynthesizerOptions { Phonemizer = new LexiconPhonemizer() };
var synthesizer = await LocalSynthesizer.LoadAsync("default", options);

var result = await synthesizer.SynthesizeAsync("Hello world.");

// A phonemizer backed by a pronunciation lexicon. A real one covers the language: a G2P model, or eSpeak NG itself.
public sealed class LexiconPhonemizer : IPhonemizer
{
    private static readonly Dictionary<string, string> Lexicon = new(StringComparer.OrdinalIgnoreCase)
    {
        ["hello"] = "həlˈəʊ",
        ["world"] = "wˈɜːld",
    };

    public ValueTask<IReadOnlyList<string>> PhonemizeAsync(
        string text, PhonemeScheme scheme, CancellationToken cancellationToken = default)
    {
        if (scheme.Type != "espeak")
            throw new NotSupportedException($"This lexicon holds eSpeak NG phonemes, not '{scheme.Type}'.");

        // One string per sentence; the full stop that ends the sentence is its last phoneme.
        var sentences = new List<string>();
        foreach (var sentence in text.Split('.', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var words = sentence.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            sentences.Add(string.Join(" ", words.Select(word => Lexicon[word])) + ".");
        }

        return new(sentences);
    }
}
```

Without a phonemizer:

| The voice's phoneme type | `SynthesizeAsync(text)` |
|--------------------------|-------------------------|
| `text` | Works. The phonemes are the characters of the text, lower-cased and decomposed (NFD), as one sentence. |
| `espeak` (every catalog voice) | Throws `NotSupportedException`, naming the voice's phoneme type and language. No audio is returned. |

## API Usage

The examples below pass phonemes; with a phonemizer set, the text methods (`SynthesizeAsync`,
`SynthesizeToFileAsync`, `SynthesizeToStreamAsync`, `SynthesizeStreamingAsync`) work the same way.

### Basic Synthesis

```csharp
// Synthesize and get result
var result = await synthesizer.SynthesizePhonemesAsync(["həlˈəʊ, wˈɜːld!"]);

// Access audio samples (float32, range -1.0 to 1.0)
float[] samples = result.AudioSamples;
int sampleRate = result.SampleRate;

// Get as byte arrays
byte[] wavBytes = result.ToWavBytes();      // WAV file format
byte[] pcmBytes = result.ToPcm16Bytes();    // Raw 16-bit PCM
```

### Save to File

```csharp
// With a phonemizer set: save directly to WAV file
await synthesizer.SynthesizeToFileAsync("Hello, world!", "output.wav");
```

### Write to Stream

```csharp
// With a phonemizer set
using var stream = new MemoryStream();
await synthesizer.SynthesizeToStreamAsync("Hello, world!", stream);

// With specific format
var options = new SynthesizeOptions { OutputFormat = AudioFormat.RawPcm16 };
await synthesizer.SynthesizeToStreamAsync("Hello, world!", stream, options);
```

### Streaming Synthesis

```csharp
// Get audio chunks as they're generated: one per sentence
string[] sentences = ["ðə wˈɛðəɹ ɪz bjˈuːtifəl tədˈeɪ.", "ðə kwˈɪk bɹˈaʊn fˈɒks dʒˈʌmps ˌəʊvə ðə lˈeɪzi dˈɒɡ."];
await foreach (var chunk in synthesizer.SynthesizePhonemesStreamingAsync(sentences))
{
    Console.WriteLine($"Chunk {chunk.Index}: {chunk.DurationSeconds:F2}s, Final: {chunk.IsFinal}");

    // Process chunk.Samples as needed
    ProcessAudioChunk(chunk.Samples, chunk.SampleRate);
}
```

### Synthesis Options

```csharp
var options = new SynthesizeOptions
{
    Speed = 1.2f,           // Faster speech (1.0 = the voice's own pace)
    SpeakerId = 0,          // Speaker ID for multi-speaker voices (0 up to the voice's speaker count)
    NoiseScale = 0.667f,    // Variability/expressiveness; unset = the voice's own default
    NoiseWidth = 0.8f,      // Duration variability; unset = the voice's own default
    OutputFormat = AudioFormat.Wav
};

var result = await synthesizer.SynthesizePhonemesAsync(["həlˈəʊ!"], options);
```

`NoiseScale` and `NoiseWidth` are `null` by default, which means the values the voice's config declares
(`inference.noise_scale`, `inference.noise_w`) — the default voice declares a noise width of 0.333, not Piper's general
0.8. `Speed` divides the length scale the config declares.

### Model Configuration

```csharp
var options = new SynthesizerOptions
{
    Provider = ExecutionProvider.Cuda,  // Use GPU
    CacheDirectory = "/custom/cache",   // Custom cache location
    ThreadCount = 4,                    // CPU threads
    DisableAutoDownload = false,        // true: load only from the cache, throw if a file is missing
    Phonemizer = null                   // the text-to-phoneme step, for voices that need one
};

var synthesizer = await LocalSynthesizer.LoadAsync("quality", options);
```

## Working with Results

### SynthesisResult

```csharp
var result = await synthesizer.SynthesizePhonemesAsync(["həlˈəʊ, wˈɜːld!"]);

// Audio data
float[] samples = result.AudioSamples;  // Raw float samples
int sampleRate = result.SampleRate;     // e.g., 22050
int channels = result.Channels;         // 1 (mono)

// Duration info
double duration = result.DurationSeconds;

// Performance metrics
Console.WriteLine($"Processing Time: {result.InferenceTimeMs:F0}ms");
Console.WriteLine($"Real-time Factor: {result.RealTimeFactor:F1}x");  // >1 = faster than real-time

// What was synthesized
Console.WriteLine($"Text: {result.Text}");                              // null when phonemes were passed
Console.WriteLine($"Phonemes: {string.Join(" | ", result.Phonemes)}");  // one string per sentence
Console.WriteLine($"Skipped: {result.UnknownPhonemes}");                // phonemes the voice has no id for
```

### Audio Conversion

```csharp
// Convert to WAV file bytes
byte[] wavBytes = result.ToWavBytes();
await File.WriteAllBytesAsync("output.wav", wavBytes);

// Convert to 16-bit PCM
byte[] pcmBytes = result.ToPcm16Bytes();
```

### AudioChunk (Streaming)

```csharp
await foreach (var chunk in synthesizer.SynthesizePhonemesStreamingAsync(sentences))
{
    // Audio samples for this chunk
    float[] samples = chunk.Samples;
    int sampleRate = chunk.SampleRate;

    // Chunk metadata
    int index = chunk.Index;            // 0-based chunk index
    bool isFinal = chunk.IsFinal;       // True for last chunk
    double duration = chunk.DurationSeconds;
}
```

## Model Selection

### Query Available Models

```csharp
// List available aliases
foreach (var alias in LocalSynthesizer.GetAvailableModels())
{
    Console.WriteLine(alias);
}

// Get detailed model information
foreach (var model in LocalSynthesizer.GetAllModels())
{
    Console.WriteLine($"{model.AliasName}: {model.DisplayName}");
    Console.WriteLine($"  Language: {model.Language}");
    Console.WriteLine($"  Voice: {model.VoiceName}");
    Console.WriteLine($"  Sample Rate: {model.SampleRate} Hz");
}
```

### Load by HuggingFace ID

```csharp
// Load specific Piper voice from HuggingFace
var synthesizer = await LocalSynthesizer.LoadAsync("rhasspy/piper-voices");
```

### Access Model Info

```csharp
var info = synthesizer.GetModelInfo();
Console.WriteLine($"Model: {info.DisplayName}");
Console.WriteLine($"Voice: {synthesizer.Voice}");
Console.WriteLine($"Sample Rate: {synthesizer.SampleRate} Hz");
Console.WriteLine($"Speakers: {info.NumSpeakers}");
```

A loaded model reports the sample rate and speaker count its config declares. Where a catalog entry says otherwise,
the config is used and `GetModelInfo()` returns the config's values.

## Audio Output Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `AudioFormat.Wav` | Standard WAV file (default) | File saving, playback |
| `AudioFormat.RawPcm16` | Raw 16-bit signed PCM | Streaming, audio APIs |
| `AudioFormat.RawFloat32` | Raw 32-bit float | Further processing |

## Performance Tips

### Model Selection

```csharp
// For real-time applications (fastest)
var synthesizer = await LocalSynthesizer.LoadAsync("fast");

// For highest quality
var synthesizer = await LocalSynthesizer.LoadAsync("quality");
```

### GPU Acceleration

```csharp
// Auto-detect best GPU
var options = new SynthesizerOptions { Provider = ExecutionProvider.Auto };

// Force specific GPU backend
var options = new SynthesizerOptions { Provider = ExecutionProvider.Cuda };     // NVIDIA
var options = new SynthesizerOptions { Provider = ExecutionProvider.CoreML };   // macOS
```

### Memory Management

```csharp
// Dispose when done
await using var synthesizer = await LocalSynthesizer.LoadAsync("default");
var result = await synthesizer.SynthesizePhonemesAsync(["həlˈəʊ!"]);

// Or explicit disposal
await synthesizer.DisposeAsync();
```

## Interface Reference

### ISynthesizerModel

```csharp
public interface ISynthesizerModel : IAsyncDisposable
{
    string? Voice { get; }
    int SampleRate { get; }
    PhonemeScheme PhonemeScheme { get; }

    Task WarmupAsync(CancellationToken cancellationToken = default);
    SynthesizerModelInfo? GetModelInfo();

    // Text: through SynthesizerOptions.Phonemizer, or built in for a voice of the "text" type
    Task<SynthesisResult> SynthesizeAsync(string text, ...);
    Task SynthesizeToStreamAsync(string text, Stream outputStream, ...);
    Task SynthesizeToFileAsync(string text, string outputPath, ...);
    IAsyncEnumerable<AudioChunk> SynthesizeStreamingAsync(string text, ...);

    // Phonemes of the voice's scheme, one string per sentence
    Task<SynthesisResult> SynthesizePhonemesAsync(IReadOnlyList<string> sentences, ...);
    IAsyncEnumerable<AudioChunk> SynthesizePhonemesStreamingAsync(IReadOnlyList<string> sentences, ...);
}
```

### IPhonemizer

```csharp
public interface IPhonemizer
{
    ValueTask<IReadOnlyList<string>> PhonemizeAsync(
        string text, PhonemeScheme scheme, CancellationToken cancellationToken = default);
}
```

## License

Piper's code is MIT, but each voice is licensed by the recordings it was trained on — read `SynthesizerModelInfo.License`
before shipping a voice:

| Alias | Voice license (from the voice's model card) |
|-------|---------------------------------------------|
| `default` | Public domain (LJ Speech) |
| `lessac` | Blizzard 2013 Lessac research license (non-commercial) |
| `fast` | CC BY-NC-SA 4.0 |
| `quality` | Unspecified |
| `british` | CC BY-NC-SA 4.0 |
| `korean` | CC BY-NC-SA 4.0 |
| `chinese` | Unknown |

Until 0.88.0 every voice was labelled MIT, which was the engine's license, not the voice's.
