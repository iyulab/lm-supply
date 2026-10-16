# LMSupply.Synthesizer

Text-to-speech synthesis using VITS/Piper models with ONNX Runtime.

English text is spoken out of the box. Every voice in the catalog was trained on eSpeak NG phonemes, and for the
English voices the package writes those phonemes itself, with a built-in English phonemizer — a pronunciation dictionary
and letter-to-sound rules, in pure .NET, offline, with no extra download (see [English text](#english-text)). Spoken by
each English voice and transcribed by Whisper, a set of ordinary sentences, sentences with figures and sentences with
words the dictionary does not have reads back at a word error rate of 1-8% (at most 3% on the plain sentences and
figures; most errors are unusual names spelled another way).

A voice in another language (the Korean and Chinese voices) speaks phonemes (`SynthesizePhonemesAsync`), or text
through an `IPhonemizer` you set; given text alone it throws `NotSupportedException` — it never speaks text as noise.

## Quick Start

```csharp
using LMSupply.Synthesizer;

// Load the default model (English US LJSpeech, public domain)
var synthesizer = await LocalSynthesizer.LoadAsync("default");

var result = await synthesizer.SynthesizeAsync("The weather is beautiful today.");
Console.WriteLine($"Duration: {result.DurationSeconds:F2}s");
Console.WriteLine($"Real-time factor: {result.RealTimeFactor:F1}x");

// Save as WAV file
await File.WriteAllBytesAsync("output.wav", result.ToWavBytes());
```

For the other languages, pass phonemes or set a phonemizer — see [Phonemes and phonemizers](#phonemes-and-phonemizers).

## Installation

```bash
dotnet add package LMSupply.Synthesizer
```

## Features

- **VITS Models**: High-quality neural TTS using Piper voices
- **The voice's own contract**: phoneme ids, inference defaults, sample rate and speaker count are read from the voice's config
- **English text out of the box**: a built-in English phonemizer that writes each English voice's own phonemes
- **Phoneme input and a phonemizer seam**: pass phonemes, or plug in the text-to-phoneme step a voice needs
- **Multiple Languages**: English (text and phonemes), Korean and Chinese (phonemes, or text through your phonemizer)
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
as `PhonemeScheme`. The five English voices speak text through the built-in English phonemizer; the Korean and Chinese
voices need phonemes or a phonemizer of yours.

## English text

The built-in English phonemizer is used for a voice of the `espeak` type whose phonemizer voice is `en`, `en-us`,
`en-gb` or `en-gb-x-rp` (or, when the config names none, whose language is English), unless
`SynthesizerOptions.Phonemizer` is set. It runs in three steps:

1. **Reading the text.** Letters are folded to plain ASCII (`café` is read as `cafe`), the text is split into sentences
   at `. ? !` and pauses are kept at `, ; :` (and at dashes and brackets). Figures are read as words: whole numbers and
   decimals (`12.5` «twelve point five»), thousands separators (`2,500`), ordinals (`21st`), a four-digit number from
   1100 to 2099 as a year (`1999` «nineteen ninety nine», `2023` «twenty twenty three»), times of day (`4:30`, `7:00`
   «seven o'clock»), dollar amounts (`$3.50` «three dollars and fifty cents», `$5 million`), percentages and a leading
   minus. A small closed set of abbreviations is expanded: `Mr. Mrs. Ms. Dr. Prof. St. vs. No. Jr. Sr. Inc. Ltd. Co. etc.
   e.g. i.e.`; a word written in capitals is spelled out letter by letter unless the dictionary reads it as a word
   (`NASA`).
2. **Pronouncing each word.** A word is looked up in the [CMU Pronouncing Dictionary](https://github.com/cmusphinx/cmudict)
   (about 126,000 words), with its stress. A word it does not have is tried as a regular ending on a word it has
   (`-s`, `-'s`, `-ed`, `-ing`, `-ly`) and as two words written as one (`smartwatch`); anything else is read by the
   letter-to-sound rules of NRL Report 7948 with a stress rule of its own. A word with two readings gets the dictionary's
   first one, whatever the sentence (`read` is always the present tense, `live` the verb); stress is each word's own,
   and the common function words (`the`, `of`, `to`, `a`, ...) are unstressed.
3. **Writing the voice's phonemes.** The dictionary is American. For `en-us` its phonemes are written directly, with
   the flapped t of «water». For the British-style `en` (the default voice) and `en-gb-x-rp` they are converted by rule:
   r is said only before a vowel (in the word, or at the start of the next one), vowels before a dropped r become long
   vowels or centring diphthongs («here» `hˈɪə`), the vowel of «lot» is told from the one of «father» by the spelling,
   and t, d and n keep the glide before a u («tune» `tjˈuːn`). What no rule can tell is left as American: the broad a of
   «bath» stays short.

Not handled, and read as the characters they are written with: dates (`2024-01-05`, `5/1`), fractions, ranges, units
(`5 km` is «five k m»), currencies other than the dollar, Roman numerals, and letters of other scripts (dropped).
Homographs are not told apart by the sentence, and names the dictionary lacks are only as good as letter-to-sound rules
are.

**Symbols the voice does not have.** The phonemizer writes only symbols the voice's own `phoneme_id_map` has an id for,
so `SynthesisResult.UnknownPhonemes` stays empty. Where a voice lacks a symbol, a fixed substitute is written: the ASCII
`g` for `ɡ`, `t` for the flap `ɾ`, `ə` for `ɚ` and for the reduced article vowel `ɐ`. Stress and length marks and pause
punctuation the voice lacks are left out. A voice that lacks any other phoneme of its accent is not guessed at: text for
it throws `NotSupportedException`, naming the missing symbols (every catalog voice has them all).

**Where the dictionary lives.** It is embedded in the `LMSupply.Synthesizer` assembly, gzip-compressed (about 0.9 MB
of the package; 3.6 MB uncompressed), so text works with no network and no cache, the moment the package is installed.
It is read on the first text a process speaks (a few hundred milliseconds) and then kept in memory (about 3 MB), shared
by every loaded voice.

**Licences.** The CMU Pronouncing Dictionary is © Carnegie Mellon University under a 2-clause BSD-style licence; its
licence text ships in the package (`licenses/cmudict/LICENSE` and `THIRD-PARTY-NOTICES.md`). The letter-to-sound rules
are from NRL Report 7948 (Naval Research Laboratory, 1976), in the public domain. No eSpeak NG code or data is used.

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

A phonemizer you set is used for every voice, the English ones included. Without one:

| The voice | `SynthesizeAsync(text)` |
|-----------|-------------------------|
| `espeak`, English (`default`, `lessac`, `fast`, `quality`, `british`) | Works, through the built-in English phonemizer ([English text](#english-text)). |
| `text` | Works. The phonemes are the characters of the text, lower-cased and decomposed (NFD), as one sentence. |
| `espeak`, another language (`korean`, `chinese`) | Throws `NotSupportedException`, naming the voice's phoneme type and language. No audio is returned. |

## API Usage

The examples below pass phonemes, which every voice takes; for an English voice, or with a phonemizer set, the text
methods (`SynthesizeAsync`, `SynthesizeToFileAsync`, `SynthesizeToStreamAsync`, `SynthesizeStreamingAsync`) work the
same way.

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
// Text, for an English voice or with a phonemizer set: save directly to WAV file
await synthesizer.SynthesizeToFileAsync("Hello, world!", "output.wav");
```

### Write to Stream

```csharp
// Text, for an English voice or with a phonemizer set
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

// Every voice of the catalog, once each
foreach (var model in LocalSynthesizer.GetAllModels())
{
    Console.WriteLine($"{model.AliasName}: {model.DisplayName}");
    Console.WriteLine($"  Key: {model.CatalogKey}");
    Console.WriteLine($"  Language: {model.Language}");
    Console.WriteLine($"  Voice: {model.VoiceName}");
    Console.WriteLine($"  Sample Rate: {model.SampleRate} Hz");
}
```

`GetAvailableModels()` returns alias names, `auto` and the aliases you registered included. `GetAllModels()` returns the
seven voices of the table above.

### Load a Voice by Its Catalog Key

The catalog's voices all live in one HuggingFace repository, `rhasspy/piper-voices`, each in a folder of its own. Their
`Id` is therefore the same, and a voice is named by its `CatalogKey`: the repository id, `:` and the voice's folder
(`VoiceName`).

```csharp
// The Lessac voice by its key, which is what the alias "lessac" resolves to
var synthesizer = await LocalSynthesizer.LoadAsync("rhasspy/piper-voices:en/en_US/lessac/medium");

// The key of a voice picked from the catalog
var korean = LocalSynthesizer.GetAllModels().First(m => m.Language == "ko-KR");
var koreanSynthesizer = await LocalSynthesizer.LoadAsync(korean.CatalogKey);
```

The repository id on its own names no voice: `LoadAsync("rhasspy/piper-voices")` loads the first voice of the catalog,
which is the default one.

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
var realTime = await LocalSynthesizer.LoadAsync("fast");

// For highest quality
var highQuality = await LocalSynthesizer.LoadAsync("quality");
```

### GPU Acceleration

```csharp
// Auto-detect best GPU
var auto = new SynthesizerOptions { Provider = ExecutionProvider.Auto };

// Force specific GPU backend
var cuda = new SynthesizerOptions { Provider = ExecutionProvider.Cuda };     // NVIDIA
var coreMl = new SynthesizerOptions { Provider = ExecutionProvider.CoreML };   // macOS
```

### Memory Management

```csharp
// Dispose when done
await using var synthesizer = await LocalSynthesizer.LoadAsync("default");
var result = await synthesizer.SynthesizePhonemesAsync(["həlˈəʊ!"]);

// Or explicit disposal
var other = await LocalSynthesizer.LoadAsync("default");
await other.DisposeAsync();
```

## Interface Reference

### ISynthesizerModel

```csharp
public interface ISynthesizerModel : IAsyncDisposable
{
    string ModelId { get; }
    bool IsGpuActive { get; }
    IReadOnlyList<string> ActiveProviders { get; }
    ExecutionProvider RequestedProvider { get; }
    long? EstimatedMemoryBytes { get; }
    string? Voice { get; }
    int SampleRate { get; }
    PhonemeScheme PhonemeScheme { get; }

    Task WarmupAsync(CancellationToken cancellationToken = default);
    SynthesizerModelInfo? GetModelInfo();

    // Text: through SynthesizerOptions.Phonemizer, or built in for a voice of the "text" type
    Task<SynthesisResult> SynthesizeAsync(string text, SynthesizeOptions? options = null, CancellationToken cancellationToken = default);
    Task SynthesizeToStreamAsync(string text, Stream outputStream, SynthesizeOptions? options = null, CancellationToken cancellationToken = default);
    Task SynthesizeToFileAsync(string text, string outputPath, SynthesizeOptions? options = null, CancellationToken cancellationToken = default);
    IAsyncEnumerable<AudioChunk> SynthesizeStreamingAsync(string text, SynthesizeOptions? options = null, CancellationToken cancellationToken = default);

    // Phonemes of the voice's scheme, one string per sentence
    Task<SynthesisResult> SynthesizePhonemesAsync(IReadOnlyList<string> sentences, SynthesizeOptions? options = null, CancellationToken cancellationToken = default);
    IAsyncEnumerable<AudioChunk> SynthesizePhonemesStreamingAsync(IReadOnlyList<string> sentences, SynthesizeOptions? options = null, CancellationToken cancellationToken = default);
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
