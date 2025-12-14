using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text.Json;
using LocalAI.Synthesizer.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LocalAI.Synthesizer.Core;

/// <summary>
/// ONNX-based implementation of VITS/Piper TTS model.
/// </summary>
internal sealed class OnnxSynthesizerModel : ISynthesizerModel
{
    private readonly SynthesizerOptions _options;
    private readonly SemaphoreSlim _lock = new(1, 1);

    private InferenceSession? _session;
    private SynthesizerModelInfo? _modelInfo;
    private VitsConfig? _config;
    private bool _isInitialized;
    private bool _isDisposed;

    public string? Voice => _modelInfo?.VoiceName;
    public int SampleRate => _config?.Audio?.SampleRate ?? 22050;

    public OnnxSynthesizerModel(SynthesizerOptions options)
    {
        _options = options.Clone();
    }

    public async Task WarmupAsync(CancellationToken cancellationToken = default)
    {
        await EnsureInitializedAsync(cancellationToken);
    }

    public SynthesizerModelInfo? GetModelInfo() => _modelInfo;

    public async Task<SynthesisResult> SynthesizeAsync(
        string text,
        SynthesizeOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        await EnsureInitializedAsync(cancellationToken);

        var sw = Stopwatch.StartNew();
        var samples = await SynthesizeCoreAsync(text, options, cancellationToken);
        sw.Stop();

        return new SynthesisResult
        {
            AudioSamples = samples,
            SampleRate = SampleRate,
            InferenceTimeMs = sw.Elapsed.TotalMilliseconds,
            Text = text
        };
    }

    public async Task SynthesizeToStreamAsync(
        string text,
        Stream outputStream,
        SynthesizeOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var result = await SynthesizeAsync(text, options, cancellationToken);
        var format = options?.OutputFormat ?? AudioFormat.Wav;

        byte[] audioData = format switch
        {
            AudioFormat.Wav => result.ToWavBytes(),
            AudioFormat.RawPcm16 => result.ToPcm16Bytes(),
            AudioFormat.RawFloat32 => FloatToBytes(result.AudioSamples),
            _ => result.ToWavBytes()
        };

        await outputStream.WriteAsync(audioData, cancellationToken);
    }

    public async Task SynthesizeToFileAsync(
        string text,
        string outputPath,
        SynthesizeOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        using var fileStream = File.Create(outputPath);
        await SynthesizeToStreamAsync(text, fileStream, options, cancellationToken);
    }

    public async IAsyncEnumerable<AudioChunk> SynthesizeStreamingAsync(
        string text,
        SynthesizeOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await EnsureInitializedAsync(cancellationToken);

        // For simplicity, split text into sentences and yield chunks
        var sentences = SplitIntoSentences(text);
        var chunkIndex = 0;

        foreach (var sentence in sentences)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (string.IsNullOrWhiteSpace(sentence))
                continue;

            var samples = await SynthesizeCoreAsync(sentence.Trim(), options, cancellationToken);

            yield return new AudioChunk
            {
                Samples = samples,
                SampleRate = SampleRate,
                Index = chunkIndex++,
                IsFinal = sentence == sentences.Last()
            };
        }
    }

    private async Task<float[]> SynthesizeCoreAsync(
        string text,
        SynthesizeOptions? options,
        CancellationToken cancellationToken)
    {
        await _lock.WaitAsync(cancellationToken);
        try
        {
            // Convert text to phoneme IDs
            var phonemeIds = TextToPhonemeIds(text);

            // Prepare inputs
            var inputIds = new DenseTensor<long>(phonemeIds, new[] { 1, phonemeIds.Length });
            var inputLengths = new DenseTensor<long>(new long[] { phonemeIds.Length }, new[] { 1 });
            var scales = new DenseTensor<float>(
                new[] { options?.NoiseScale ?? 0.667f, options?.Speed ?? 1.0f, options?.NoiseWidth ?? 0.8f },
                new[] { 3 });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputIds),
                NamedOnnxValue.CreateFromTensor("input_lengths", inputLengths),
                NamedOnnxValue.CreateFromTensor("scales", scales)
            };

            // Add speaker ID for multi-speaker models
            if (_modelInfo?.NumSpeakers > 1)
            {
                var speakerId = new DenseTensor<long>(new long[] { options?.SpeakerId ?? 0 }, new[] { 1 });
                inputs.Add(NamedOnnxValue.CreateFromTensor("sid", speakerId));
            }

            // Run inference
            using var results = _session!.Run(inputs);
            var output = results.First().AsTensor<float>();

            return output.ToArray();
        }
        finally
        {
            _lock.Release();
        }
    }

    private long[] TextToPhonemeIds(string text)
    {
        // Simplified phoneme conversion - in production, use proper G2P (grapheme-to-phoneme)
        // This is a basic character-to-ID mapping for demonstration
        var ids = new List<long> { 0 }; // Start token

        foreach (var c in text.ToLower())
        {
            var id = c switch
            {
                ' ' => 1,
                'a' => 2,
                'b' => 3,
                'c' => 4,
                'd' => 5,
                'e' => 6,
                'f' => 7,
                'g' => 8,
                'h' => 9,
                'i' => 10,
                'j' => 11,
                'k' => 12,
                'l' => 13,
                'm' => 14,
                'n' => 15,
                'o' => 16,
                'p' => 17,
                'q' => 18,
                'r' => 19,
                's' => 20,
                't' => 21,
                'u' => 22,
                'v' => 23,
                'w' => 24,
                'x' => 25,
                'y' => 26,
                'z' => 27,
                '.' => 28,
                ',' => 29,
                '!' => 30,
                '?' => 31,
                '\'' => 32,
                '-' => 33,
                _ => 1 // Default to space for unknown
            };
            ids.Add(id);
        }

        ids.Add(0); // End token
        return ids.ToArray();
    }

    private static string[] SplitIntoSentences(string text)
    {
        // Simple sentence splitting
        return text.Split(['.', '!', '?'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
    }

    private async Task EnsureInitializedAsync(CancellationToken cancellationToken)
    {
        if (_isInitialized) return;

        await _lock.WaitAsync(cancellationToken);
        try
        {
            if (_isInitialized) return;

            // Resolve model info
            if (!SynthesizerModelRegistry.Default.TryGet(_options.ModelId, out _modelInfo))
            {
                // Treat as HuggingFace model ID
                _modelInfo = new SynthesizerModelInfo
                {
                    Id = _options.ModelId,
                    Alias = _options.ModelId,
                    DisplayName = _options.ModelId,
                    Architecture = "VITS"
                };
            }

            // Download model if needed
            var modelPath = await ResolveModelPathAsync(cancellationToken);

            // Load config
            var configPath = Path.Combine(modelPath, _modelInfo!.ConfigFile);
            if (File.Exists(configPath))
            {
                var configJson = await File.ReadAllTextAsync(configPath, cancellationToken);
                _config = JsonSerializer.Deserialize<VitsConfig>(configJson, new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });
            }

            // Load model
            var modelFilePath = Path.Combine(modelPath, _modelInfo.ModelFile);
            if (!File.Exists(modelFilePath))
            {
                throw new FileNotFoundException($"Model file not found: {modelFilePath}");
            }

            _session = await OnnxSessionFactory.CreateAsync(
                modelFilePath,
                _options.Provider,
                ConfigureSessionOptions,
                cancellationToken: cancellationToken);

            _isInitialized = true;
        }
        finally
        {
            _lock.Release();
        }
    }

    private async Task<string> ResolveModelPathAsync(CancellationToken cancellationToken)
    {
        // If it's a local directory path, return it
        if (Directory.Exists(_modelInfo!.Id))
        {
            return _modelInfo.Id;
        }

        // Check if parent directory exists (for file paths)
        var parentDir = Path.GetDirectoryName(_modelInfo.Id);
        if (parentDir != null && Directory.Exists(parentDir))
        {
            return parentDir;
        }

        // Download from HuggingFace
        var cacheDir = _options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        using var downloader = new HuggingFaceDownloader(cacheDir);

        var modelPath = await downloader.DownloadModelAsync(
            _modelInfo.Id,
            subfolder: _modelInfo.VoiceName,
            cancellationToken: cancellationToken);

        return modelPath;
    }

    private void ConfigureSessionOptions(SessionOptions options)
    {
        if (_options.ThreadCount.HasValue)
        {
            options.IntraOpNumThreads = _options.ThreadCount.Value;
            options.InterOpNumThreads = _options.ThreadCount.Value;
        }
    }

    private static byte[] FloatToBytes(float[] samples)
    {
        var bytes = new byte[samples.Length * 4];
        Buffer.BlockCopy(samples, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    public void Dispose()
    {
        if (_isDisposed) return;
        _isDisposed = true;

        _session?.Dispose();
        _lock.Dispose();
    }

    public ValueTask DisposeAsync()
    {
        Dispose();
        return ValueTask.CompletedTask;
    }
}

/// <summary>
/// VITS model configuration from JSON.
/// </summary>
internal sealed class VitsConfig
{
    public AudioConfig? Audio { get; set; }
    public int NumSymbols { get; set; }
    public int NumSpeakers { get; set; }
}

/// <summary>
/// Audio configuration.
/// </summary>
internal sealed class AudioConfig
{
    public int SampleRate { get; set; } = 22050;
}
