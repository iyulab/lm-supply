# LocalAI.Transcriber

Speech-to-text transcription using Whisper models with ONNX Runtime.

## Quick Start

```csharp
using LocalAI.Transcriber;

// Load the default model (Whisper Base)
var transcriber = await LocalTranscriber.LoadAsync("default");

// Transcribe audio file
var result = await transcriber.TranscribeAsync("audio.wav");
Console.WriteLine(result.Text);
Console.WriteLine($"Language: {result.Language}");
Console.WriteLine($"Duration: {result.DurationSeconds}s");
```

## Installation

```bash
dotnet add package LocalAI.Transcriber
```

## Features

- **Whisper Models**: OpenAI Whisper models optimized for ONNX Runtime
- **Multiple Sizes**: From tiny (39M) to large-v3 (1.5B) parameters
- **Multilingual**: Support for 99+ languages with auto-detection
- **Timestamps**: Word-level and segment-level timestamps
- **Streaming**: Real-time transcription as audio is processed
- **GPU Acceleration**: CUDA, DirectML, and CoreML support

## Available Models

| Alias | Model | Parameters | Size | WER | Description |
|-------|-------|------------|------|-----|-------------|
| `fast` | Whisper Tiny | 39M | ~150MB | 7.6% | Ultra-fast, lowest accuracy |
| `default` | Whisper Base | 74M | ~290MB | 5.0% | Balanced speed and accuracy |
| `quality` | Whisper Small | 244M | ~970MB | 3.4% | Higher accuracy |
| `medium` | Whisper Medium | 769M | ~3GB | 2.9% | High quality |
| `large` | Whisper Large V3 | 1.5B | ~6GB | 2.5% | Highest accuracy |
| `english` | Whisper Base.en | 74M | ~290MB | 4.3% | Optimized for English |

## API Usage

### Basic Transcription

```csharp
// Transcribe from file
var result = await transcriber.TranscribeAsync("audio.wav");

// Transcribe from stream
using var stream = File.OpenRead("audio.wav");
var result = await transcriber.TranscribeAsync(stream);

// Transcribe from byte array
var audioData = await File.ReadAllBytesAsync("audio.wav");
var result = await transcriber.TranscribeAsync(audioData);
```

### Streaming Transcription

```csharp
// Get segments as they're processed
await foreach (var segment in transcriber.TranscribeStreamingAsync("audio.wav"))
{
    Console.WriteLine($"[{segment.Start:F2}s - {segment.End:F2}s] {segment.Text}");
}
```

### Transcription Options

```csharp
var options = new TranscribeOptions
{
    Language = "en",        // Force specific language (null for auto-detect)
    Translate = true,       // Translate to English
    WordTimestamps = true,  // Include word-level timestamps
    InitialPrompt = "Technical discussion about AI" // Guide transcription style
};

var result = await transcriber.TranscribeAsync("audio.wav", options);
```

### Model Configuration

```csharp
var options = new TranscriberOptions
{
    Provider = ExecutionProvider.Cuda,  // Use GPU
    CacheDirectory = "/custom/cache",   // Custom cache location
    ThreadCount = 4                     // CPU threads
};

var transcriber = await LocalTranscriber.LoadAsync("quality", options);
```

## Working with Results

### TranscriptionResult

```csharp
var result = await transcriber.TranscribeAsync("audio.wav");

// Full transcribed text
Console.WriteLine(result.Text);

// Detected language with confidence
Console.WriteLine($"Language: {result.Language} ({result.LanguageProbability:P0})");

// Audio and processing metrics
Console.WriteLine($"Audio Duration: {result.DurationSeconds:F2}s");
Console.WriteLine($"Processing Time: {result.InferenceTimeMs:F0}ms");
Console.WriteLine($"Real-time Factor: {result.RealTimeFactor:F1}x");
```

### Working with Segments

```csharp
foreach (var segment in result.Segments)
{
    // Formatted timestamp output
    Console.WriteLine(segment.ToString());
    // Output: [00:05.50 --> 00:08.30] Hello, welcome to our podcast.

    // Access segment properties
    Console.WriteLine($"ID: {segment.Id}");
    Console.WriteLine($"Start: {segment.Start}s");
    Console.WriteLine($"End: {segment.End}s");
    Console.WriteLine($"Duration: {segment.Duration}s");
    Console.WriteLine($"Text: {segment.Text}");

    // Quality metrics
    Console.WriteLine($"Avg Log Prob: {segment.AvgLogProb}");
    Console.WriteLine($"No Speech Prob: {segment.NoSpeechProb}");
}
```

### Word-Level Timestamps

```csharp
var options = new TranscribeOptions { WordTimestamps = true };
var result = await transcriber.TranscribeAsync("audio.wav", options);

foreach (var segment in result.Segments)
{
    if (segment.Words != null)
    {
        foreach (var word in segment.Words)
        {
            Console.WriteLine($"[{word.Start:F2}s] {word.Word} ({word.Probability:P0})");
        }
    }
}
```

## Model Selection

### Query Available Models

```csharp
// List available aliases
foreach (var alias in LocalTranscriber.GetAvailableModels())
{
    Console.WriteLine(alias);
}

// Get detailed model information
foreach (var model in LocalTranscriber.GetAllModels())
{
    Console.WriteLine($"{model.Alias}: {model.DisplayName}");
    Console.WriteLine($"  Parameters: {model.ParametersM}M");
    Console.WriteLine($"  WER: {model.WerLibriSpeech}%");
    Console.WriteLine($"  Multilingual: {model.IsMultilingual}");
}
```

### Load by HuggingFace ID

```csharp
// Load specific ONNX model from HuggingFace
var transcriber = await LocalTranscriber.LoadAsync("onnx-community/whisper-small");
```

## Audio Format Support

Supported audio formats (via NAudio):
- WAV (recommended)
- MP3
- AAC/M4A
- FLAC
- OGG

Audio is automatically:
- Resampled to 16kHz
- Converted to mono
- Normalized to float samples

## Performance Tips

### Model Selection

```csharp
// For real-time applications (fast)
var transcriber = await LocalTranscriber.LoadAsync("fast");

// For batch processing (quality)
var transcriber = await LocalTranscriber.LoadAsync("quality");

// For maximum accuracy (large)
var transcriber = await LocalTranscriber.LoadAsync("large");
```

### GPU Acceleration

```csharp
// Auto-detect best GPU
var options = new TranscriberOptions { Provider = ExecutionProvider.Auto };

// Force specific GPU backend
var options = new TranscriberOptions { Provider = ExecutionProvider.Cuda };     // NVIDIA
var options = new TranscriberOptions { Provider = ExecutionProvider.DirectML }; // Windows/AMD
var options = new TranscriberOptions { Provider = ExecutionProvider.CoreML };   // macOS
```

### Memory Management

```csharp
// Dispose when done
using var transcriber = await LocalTranscriber.LoadAsync("default");
var result = await transcriber.TranscribeAsync("audio.wav");

// Or explicit disposal
await transcriber.DisposeAsync();
```

## Interface Reference

### ITranscriberModel

```csharp
public interface ITranscriberModel : IDisposable, IAsyncDisposable
{
    string? Language { get; }
    Task WarmupAsync(CancellationToken cancellationToken = default);
    TranscriberModelInfo? GetModelInfo();

    Task<TranscriptionResult> TranscribeAsync(string audioPath, ...);
    Task<TranscriptionResult> TranscribeAsync(Stream audioStream, ...);
    Task<TranscriptionResult> TranscribeAsync(byte[] audioData, ...);
    IAsyncEnumerable<TranscriptionSegment> TranscribeStreamingAsync(string audioPath, ...);
}
```

## License

All default Whisper models are MIT licensed.
