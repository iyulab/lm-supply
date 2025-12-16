# LMSupply.Synthesizer

Text-to-speech synthesis using VITS/Piper models with ONNX Runtime.

## Quick Start

```csharp
using LMSupply.Synthesizer;

// Load the default model (English US Lessac)
var synthesizer = await LocalSynthesizer.LoadAsync("default");

// Synthesize speech from text
var result = await synthesizer.SynthesizeAsync("Hello, welcome to LMSupply!");
Console.WriteLine($"Duration: {result.DurationSeconds:F2}s");
Console.WriteLine($"Real-time factor: {result.RealTimeFactor:F1}x");

// Save as WAV file
await synthesizer.SynthesizeToFileAsync("Hello world!", "output.wav");
```

## Installation

```bash
dotnet add package LMSupply.Synthesizer
```

## Features

- **VITS Models**: High-quality neural TTS using Piper voices
- **Multiple Languages**: English, Korean, Japanese, Chinese, and more
- **Fast Inference**: Optimized ONNX models for real-time synthesis
- **Streaming**: Generate audio chunks progressively
- **Multiple Formats**: WAV, raw PCM16, and float32 output
- **GPU Acceleration**: CUDA, DirectML, and CoreML support

## Available Models

| Alias | Voice | Language | Sample Rate | Size | Description |
|-------|-------|----------|-------------|------|-------------|
| `default` | Lessac | en-US | 22050 Hz | ~64MB | High-quality US English female |
| `fast` | Ryan | en-US | 16000 Hz | ~16MB | Fast US English male |
| `quality` | Amy | en-US | 22050 Hz | ~64MB | High-quality US English female |
| `british` | Semaine | en-GB | 22050 Hz | ~64MB | British English female |
| `korean` | KSS | ko-KR | 22050 Hz | ~64MB | Korean female |
| `japanese` | JSUT | ja-JP | 22050 Hz | ~64MB | Japanese female |
| `chinese` | Huayan | zh-CN | 22050 Hz | ~64MB | Mandarin Chinese female |

## API Usage

### Basic Synthesis

```csharp
// Synthesize and get result
var result = await synthesizer.SynthesizeAsync("Hello, world!");

// Access audio samples (float32, range -1.0 to 1.0)
float[] samples = result.AudioSamples;
int sampleRate = result.SampleRate;

// Get as byte arrays
byte[] wavBytes = result.ToWavBytes();      // WAV file format
byte[] pcmBytes = result.ToPcm16Bytes();    // Raw 16-bit PCM
```

### Save to File

```csharp
// Save directly to WAV file
await synthesizer.SynthesizeToFileAsync("Hello, world!", "output.wav");
```

### Write to Stream

```csharp
using var stream = new MemoryStream();
await synthesizer.SynthesizeToStreamAsync("Hello, world!", stream);

// With specific format
var options = new SynthesizeOptions { OutputFormat = AudioFormat.RawPcm16 };
await synthesizer.SynthesizeToStreamAsync("Hello, world!", stream, options);
```

### Streaming Synthesis

```csharp
// Get audio chunks as they're generated (by sentence)
await foreach (var chunk in synthesizer.SynthesizeStreamingAsync("Hello. How are you today?"))
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
    Speed = 1.2f,           // Faster speech (1.0 = normal)
    Pitch = 0.0f,           // Pitch shift in semitones
    SpeakerId = 0,          // Speaker ID for multi-speaker models
    NoiseScale = 0.667f,    // Variability/expressiveness
    NoiseWidth = 0.8f,      // Duration variability
    OutputFormat = AudioFormat.Wav
};

var result = await synthesizer.SynthesizeAsync("Hello!", options);
```

### Model Configuration

```csharp
var options = new SynthesizerOptions
{
    Provider = ExecutionProvider.Cuda,  // Use GPU
    CacheDirectory = "/custom/cache",   // Custom cache location
    ThreadCount = 4                     // CPU threads
};

var synthesizer = await LocalSynthesizer.LoadAsync("quality", options);
```

## Working with Results

### SynthesisResult

```csharp
var result = await synthesizer.SynthesizeAsync("Hello, world!");

// Audio data
float[] samples = result.AudioSamples;  // Raw float samples
int sampleRate = result.SampleRate;     // e.g., 22050
int channels = result.Channels;         // 1 (mono)

// Duration info
double duration = result.DurationSeconds;

// Performance metrics
Console.WriteLine($"Processing Time: {result.InferenceTimeMs:F0}ms");
Console.WriteLine($"Real-time Factor: {result.RealTimeFactor:F1}x");  // >1 = faster than real-time

// Original text
Console.WriteLine($"Text: {result.Text}");
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
await foreach (var chunk in synthesizer.SynthesizeStreamingAsync(text))
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
    Console.WriteLine($"{model.Alias}: {model.DisplayName}");
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
```

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
var options = new SynthesizerOptions { Provider = ExecutionProvider.DirectML }; // Windows/AMD
var options = new SynthesizerOptions { Provider = ExecutionProvider.CoreML };   // macOS
```

### Memory Management

```csharp
// Dispose when done
using var synthesizer = await LocalSynthesizer.LoadAsync("default");
var result = await synthesizer.SynthesizeAsync("Hello!");

// Or explicit disposal
await synthesizer.DisposeAsync();
```

## Interface Reference

### ISynthesizerModel

```csharp
public interface ISynthesizerModel : IDisposable, IAsyncDisposable
{
    string? Voice { get; }
    int SampleRate { get; }

    Task WarmupAsync(CancellationToken cancellationToken = default);
    SynthesizerModelInfo? GetModelInfo();

    Task<SynthesisResult> SynthesizeAsync(string text, ...);
    Task SynthesizeToStreamAsync(string text, Stream outputStream, ...);
    Task SynthesizeToFileAsync(string text, string outputPath, ...);
    IAsyncEnumerable<AudioChunk> SynthesizeStreamingAsync(string text, ...);
}
```

## License

All default Piper voices are MIT licensed.
