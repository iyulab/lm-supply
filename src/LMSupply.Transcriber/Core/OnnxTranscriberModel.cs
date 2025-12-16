using System.Diagnostics;
using System.Runtime.CompilerServices;
using LMSupply.Transcriber.Audio;
using LMSupply.Transcriber.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.Transcriber.Core;

/// <summary>
/// ONNX-based implementation of Whisper transcription model.
/// </summary>
internal sealed class OnnxTranscriberModel : ITranscriberModel
{
    private readonly TranscriberOptions _options;
    private readonly SemaphoreSlim _lock = new(1, 1);

    private InferenceSession? _encoderSession;
    private InferenceSession? _decoderSession;
    private TranscriberModelInfo? _modelInfo;
    private bool _isInitialized;
    private bool _isDisposed;

    /// <inheritdoc />
    public string ModelId => _modelInfo?.Id ?? _options.ModelId;

    public string? Language => null; // Auto-detected per transcription

    public OnnxTranscriberModel(TranscriberOptions options)
    {
        _options = options.Clone();
    }

    public async Task WarmupAsync(CancellationToken cancellationToken = default)
    {
        await EnsureInitializedAsync(cancellationToken);
    }

    public TranscriberModelInfo? GetModelInfo() => _modelInfo;

    public async Task<TranscriptionResult> TranscribeAsync(
        string audioPath,
        TranscribeOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var samples = await AudioProcessor.LoadAudioAsync(audioPath, cancellationToken);
        return await TranscribeCoreAsync(samples, options, cancellationToken);
    }

    public async Task<TranscriptionResult> TranscribeAsync(
        Stream audioStream,
        TranscribeOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var samples = await AudioProcessor.LoadAudioAsync(audioStream, cancellationToken);
        return await TranscribeCoreAsync(samples, options, cancellationToken);
    }

    public async Task<TranscriptionResult> TranscribeAsync(
        byte[] audioData,
        TranscribeOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var samples = await AudioProcessor.LoadAudioAsync(audioData, cancellationToken);
        return await TranscribeCoreAsync(samples, options, cancellationToken);
    }

    public async IAsyncEnumerable<TranscriptionSegment> TranscribeStreamingAsync(
        string audioPath,
        TranscribeOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await EnsureInitializedAsync(cancellationToken);
        var samples = await AudioProcessor.LoadAudioAsync(audioPath, cancellationToken);

        var chunks = AudioProcessor.SplitIntoChunks(samples);
        var segmentId = 0;

        foreach (var chunk in chunks)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var chunkStartTime = segmentId * 30.0;
            var result = await TranscribeChunkAsync(chunk, options, cancellationToken);

            foreach (var segment in result.Segments)
            {
                yield return new TranscriptionSegment
                {
                    Id = segmentId++,
                    Start = chunkStartTime + segment.Start,
                    End = chunkStartTime + segment.End,
                    Text = segment.Text,
                    AvgLogProb = segment.AvgLogProb,
                    NoSpeechProb = segment.NoSpeechProb,
                    CompressionRatio = segment.CompressionRatio
                };
            }
        }
    }

    private async Task<TranscriptionResult> TranscribeCoreAsync(
        float[] samples,
        TranscribeOptions? options,
        CancellationToken cancellationToken)
    {
        await EnsureInitializedAsync(cancellationToken);

        var sw = Stopwatch.StartNew();
        var duration = AudioProcessor.GetDurationSeconds(samples);

        // For short audio, process as single chunk
        if (samples.Length <= 480000) // 30 seconds
        {
            var result = await TranscribeChunkAsync(samples, options, cancellationToken);
            sw.Stop();

            return new TranscriptionResult
            {
                Text = result.Text,
                Language = result.Language,
                LanguageProbability = result.LanguageProbability,
                Segments = result.Segments,
                DurationSeconds = duration,
                InferenceTimeMs = sw.Elapsed.TotalMilliseconds
            };
        }

        // For longer audio, process in chunks
        var chunks = AudioProcessor.SplitIntoChunks(samples);
        var allSegments = new List<TranscriptionSegment>();
        var textParts = new List<string>();
        string? detectedLanguage = null;
        float? languageProb = null;

        for (int i = 0; i < chunks.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var chunkResult = await TranscribeChunkAsync(chunks[i], options, cancellationToken);
            var chunkStartTime = i * 30.0;

            if (i == 0)
            {
                detectedLanguage = chunkResult.Language;
                languageProb = chunkResult.LanguageProbability;
            }

            textParts.Add(chunkResult.Text);

            foreach (var segment in chunkResult.Segments)
            {
                allSegments.Add(new TranscriptionSegment
                {
                    Id = allSegments.Count,
                    Start = chunkStartTime + segment.Start,
                    End = chunkStartTime + segment.End,
                    Text = segment.Text,
                    AvgLogProb = segment.AvgLogProb,
                    NoSpeechProb = segment.NoSpeechProb,
                    CompressionRatio = segment.CompressionRatio
                });
            }
        }

        sw.Stop();

        return new TranscriptionResult
        {
            Text = string.Join(" ", textParts),
            Language = detectedLanguage ?? "en",
            LanguageProbability = languageProb,
            Segments = allSegments,
            DurationSeconds = duration,
            InferenceTimeMs = sw.Elapsed.TotalMilliseconds
        };
    }

    private async Task<TranscriptionResult> TranscribeChunkAsync(
        float[] samples,
        TranscribeOptions? options,
        CancellationToken cancellationToken)
    {
        await _lock.WaitAsync(cancellationToken);
        try
        {
            // Compute mel spectrogram
            var numMelBins = _modelInfo?.NumMelBins ?? 80;
            var melSpec = AudioProcessor.ComputeLogMelSpectrogram(samples, numMelBins);

            // Run encoder
            var encoderOutput = await RunEncoderAsync(melSpec, numMelBins, cancellationToken);

            // Run decoder with greedy decoding
            var (text, segments) = await RunDecoderAsync(encoderOutput, options, cancellationToken);

            return new TranscriptionResult
            {
                Text = text,
                Language = options?.Language ?? "en", // TODO: Implement language detection
                LanguageProbability = null,
                Segments = segments
            };
        }
        finally
        {
            _lock.Release();
        }
    }

    private Task<float[]> RunEncoderAsync(float[] melSpec, int numMelBins, CancellationToken cancellationToken)
    {
        return Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();

            var inputTensor = new DenseTensor<float>(melSpec, [1, numMelBins, 3000]);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_features", inputTensor)
            };

            using var results = _encoderSession!.Run(inputs);
            var output = results.First().AsTensor<float>();

            // Copy tensor output to array
            return output.ToArray();
        }, cancellationToken);
    }

    private Task<(string text, List<TranscriptionSegment> segments)> RunDecoderAsync(
        float[] encoderOutput,
        TranscribeOptions? options,
        CancellationToken cancellationToken)
    {
        return Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Simplified greedy decoding
            // In production, implement proper beam search and token decoding

            var segments = new List<TranscriptionSegment>
            {
                new()
                {
                    Id = 0,
                    Start = 0,
                    End = 30,
                    Text = "[Transcription placeholder - implement proper decoder]"
                }
            };

            return ("[Transcription placeholder - implement proper decoder]", segments);
        }, cancellationToken);
    }

    private async Task EnsureInitializedAsync(CancellationToken cancellationToken)
    {
        if (_isInitialized) return;

        await _lock.WaitAsync(cancellationToken);
        try
        {
            if (_isInitialized) return;

            // Resolve model info
            if (!TranscriberModelRegistry.Default.TryGet(_options.ModelId, out _modelInfo))
            {
                // Treat as HuggingFace model ID
                _modelInfo = new TranscriberModelInfo
                {
                    Id = _options.ModelId,
                    Alias = _options.ModelId,
                    DisplayName = _options.ModelId,
                    Architecture = "Whisper"
                };
            }

            // Download model if needed
            var modelPath = await ResolveModelPathAsync(cancellationToken);

            // Load encoder
            var encoderPath = Path.Combine(modelPath, _modelInfo!.EncoderFile);
            if (!File.Exists(encoderPath))
            {
                throw new FileNotFoundException($"Encoder model not found: {encoderPath}");
            }

            _encoderSession = await OnnxSessionFactory.CreateAsync(
                encoderPath,
                _options.Provider,
                ConfigureSessionOptions,
                cancellationToken: cancellationToken);

            // Load decoder if available
            var decoderPath = Path.Combine(modelPath, _modelInfo.DecoderFile);
            if (File.Exists(decoderPath))
            {
                _decoderSession = await OnnxSessionFactory.CreateAsync(
                    decoderPath,
                    _options.Provider,
                    ConfigureSessionOptions,
                    cancellationToken: cancellationToken);
            }

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
            subfolder: "onnx",
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

    public ValueTask DisposeAsync()
    {
        if (_isDisposed) return ValueTask.CompletedTask;
        _isDisposed = true;

        _encoderSession?.Dispose();
        _decoderSession?.Dispose();
        _lock.Dispose();
        return ValueTask.CompletedTask;
    }
}
