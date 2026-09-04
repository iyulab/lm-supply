using System.Diagnostics;
using System.Runtime.CompilerServices;
using LMSupply.Core.Download;
using LMSupply.Download;
using LMSupply.Inference;
using LMSupply.Transcriber.Audio;
using LMSupply.Transcriber.Decoding;
using LMSupply.Transcriber.Internal;
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
    private SessionCreationResult? _encoderSessionInfo;
    private WhisperTokenizer? _tokenizer;
    private WhisperDecoder? _decoder;
    private TranscriberModelInfo? _modelInfo;
    private string? _modelPath;
    private bool _isInitialized;
    private bool _isDisposed;

    /// <inheritdoc />
    public string ModelId => _modelInfo?.Id ?? _options.ModelId;

    public string? Language => null; // Auto-detected per transcription

    /// <summary>
    /// Gets whether GPU acceleration is actually being used for inference.
    /// </summary>
    public bool IsGpuActive => _encoderSessionInfo?.IsGpuActive ?? false;

    /// <summary>
    /// Gets the list of active execution providers for the encoder session.
    /// </summary>
    public IReadOnlyList<string> ActiveProviders => _encoderSessionInfo?.ActiveProviders ?? [];

    /// <summary>
    /// Gets the execution provider that was requested.
    /// </summary>
    public ExecutionProvider RequestedProvider => _encoderSessionInfo?.RequestedProvider ?? ExecutionProvider.Auto;

    /// <inheritdoc />
    public long? EstimatedMemoryBytes => _modelInfo?.SizeBytes * 2;

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
        ValidateTranslateSupport(options);
        var samples = await AudioProcessor.LoadAudioAsync(audioPath, cancellationToken);

        var chunks = AudioProcessor.SplitIntoChunks(samples);
        var segmentId = 0;
        string? lastYieldedText = null;
        var compressionThreshold = options?.CompressionRatioThreshold ?? 2.4f;
        var noSpeechThreshold = options?.NoSpeechThreshold ?? 0.6f;

        for (int i = 0; i < chunks.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var chunkStartTime = i * 30.0;
            var result = await TranscribeChunkAsync(chunks[i], options, cancellationToken);

            foreach (var segment in result.Segments)
            {
                var trimmedText = segment.Text.Trim();

                // Skip consecutive duplicates across chunks
                if (string.Equals(trimmedText, lastYieldedText, StringComparison.Ordinal))
                    continue;

                // Skip high compression ratio (hallucination)
                var ratio = segment.CompressionRatio
                    ?? SegmentPostProcessor.ComputeCompressionRatio(trimmedText);
                if (ratio > compressionThreshold)
                    continue;

                // Skip high no-speech probability
                if (segment.NoSpeechProb.HasValue && segment.NoSpeechProb.Value > noSpeechThreshold)
                    continue;

                lastYieldedText = trimmedText;
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

    /// <summary>
    /// Validates that the loaded model supports the translate task when requested.
    /// Fails fast with a clear message instead of silently producing a source-language transcript.
    /// </summary>
    private void ValidateTranslateSupport(TranscribeOptions? options)
    {
        if (options?.Translate != true)
            return;

        if (_modelInfo is { IsTranslateSupported: false })
        {
            throw new NotSupportedException(
                $"Model '{_modelInfo.Id}' is English-only and does not support the Whisper translate task. " +
                $"Whisper translate (speech → English text) requires a multilingual model. " +
                $"Use an alias such as 'default', 'quality', 'large', or 'turbo' instead.");
        }
    }

    private async Task<TranscriptionResult> TranscribeCoreAsync(
        float[] samples,
        TranscribeOptions? options,
        CancellationToken cancellationToken)
    {
        await EnsureInitializedAsync(cancellationToken);
        ValidateTranslateSupport(options);

        var sw = Stopwatch.StartNew();
        var duration = AudioProcessor.GetDurationSeconds(samples);

        // For short audio, process as single chunk
        if (samples.Length <= 480000) // 30 seconds
        {
            var result = await TranscribeChunkAsync(samples, options, cancellationToken);
            var (filteredSegments, filteredText) = SegmentPostProcessor.Process(
                result.Segments.ToList(), options);
            sw.Stop();

            return new TranscriptionResult
            {
                Text = filteredText,
                Language = result.Language,
                LanguageProbability = result.LanguageProbability,
                Segments = filteredSegments,
                DurationSeconds = duration,
                InferenceTimeMs = sw.Elapsed.TotalMilliseconds
            };
        }

        // For longer audio, process in fixed 30-second chunks.
        // With WordTimestamps enabled, the decoder produces natural segment boundaries
        // within each chunk. Fixed chunking preserves performance (no redundant encoding).
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

        // Apply post-processing: dedup + threshold filtering
        var (postSegments, postText) = SegmentPostProcessor.Process(allSegments, options);
        sw.Stop();

        return new TranscriptionResult
        {
            Text = postText,
            Language = detectedLanguage ?? "en",
            LanguageProbability = languageProb,
            Segments = postSegments,
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
            // Log language settings for debugging
            Trace.TraceInformation($"[OnnxTranscriberModel] Transcribing chunk - Language: {options?.Language ?? "auto-detect"}, " +
                $"WordTimestamps: {options?.WordTimestamps ?? false}");

            // Compute mel spectrogram
            var numMelBins = _modelInfo?.NumMelBins ?? 80;
            var melSpec = AudioProcessor.ComputeLogMelSpectrogram(samples, numMelBins);

            // Run encoder
            var encoderOutput = await RunEncoderAsync(melSpec, numMelBins, cancellationToken);

            // Audio is always padded/truncated to a fixed 30s window before mel extraction
            // (AudioProcessor.PadOrTruncate), so the encoder's own sequence length can't tell us
            // how long this chunk's real content was. Compute it from the pre-padding sample
            // count instead, clamped to the padding window.
            var sampleRate = _modelInfo?.SampleRate ?? 16000;
            var chunkDurationSeconds = Math.Min(samples.Length / (double)sampleRate, 30.0);

            // Run decoder with greedy decoding (includes language detection)
            var decoderResult = await RunDecoderAsync(encoderOutput, chunkDurationSeconds, options, cancellationToken);

            Trace.TraceInformation($"[OnnxTranscriberModel] Transcription result - Language: {decoderResult.Language}, " +
                $"Probability: {decoderResult.LanguageProbability?.ToString("F3", System.Globalization.CultureInfo.InvariantCulture) ?? "N/A"}, " +
                $"Text length: {decoderResult.Text.Length}, Segments: {decoderResult.Segments.Count}");

            return new TranscriptionResult
            {
                Text = decoderResult.Text,
                Language = decoderResult.Language,
                LanguageProbability = decoderResult.LanguageProbability,
                Segments = decoderResult.Segments
            };
        }
        finally
        {
            _lock.Release();
        }
    }

    private Task<float[]> RunEncoderAsync(float[] melSpec, int numMelBins, CancellationToken cancellationToken)
    {
        // CancellableInference guarantees control returns to the caller if the token is
        // cancelled, or after a bounded default timeout, even when the native ONNX call (e.g. a
        // cold DirectML kernel init) ignores cancellation and blocks indefinitely.
        return CancellableInference.RunAsync(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();

            var inputTensor = new DenseTensor<float>(melSpec, [1, numMelBins, 3000]);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_features", inputTensor)
            };

            using var results = _encoderSession!.Run(inputs);
            var output = results[0].AsTensor<float>();

            // Copy tensor output to array
            return output.ToArray();
        }, cancellationToken);
    }

    private async Task<DecodingResult> RunDecoderAsync(
        float[] encoderOutput,
        double chunkDurationSeconds,
        TranscribeOptions? options,
        CancellationToken cancellationToken)
    {
        if (_decoder == null)
        {
            throw new InvalidOperationException("Decoder not initialized");
        }

        // Get encoder output dimensions
        // Whisper encoder output shape: [1, sequence_length, hidden_size]
        // sequence_length = 1500 (for 30s audio), hidden_size = d_model from config
        var hiddenSize = _modelInfo?.HiddenSize ?? 512; // Default for base model
        var sequenceLength = encoderOutput.Length / hiddenSize;

        return await _decoder.DecodeAsync(
            encoderOutput,
            sequenceLength,
            hiddenSize,
            chunkDurationSeconds,
            options,
            cancellationToken);
    }

    private async Task EnsureInitializedAsync(CancellationToken cancellationToken)
    {
        if (_isInitialized) return;

        await _lock.WaitAsync(cancellationToken);
        try
        {
            if (_isInitialized) return;

            // Resolve model info
            _modelInfo = TranscriberModelRegistry.Default.Resolve(_options.ModelId);

            // Download model if needed and get discovery result
            var (baseModelPath, discovery) = await ResolveModelPathAsync(cancellationToken);

            // Auto-detect model parameters from config.json for fallback models
            // Fallback models have AliasName == Id (set by CreateFallbackModelInfo)
            if (_modelInfo!.AliasName == _modelInfo.Id)
            {
                var configDir = discovery?.GetOnnxDirectory(baseModelPath);

                // Try onnx subdirectory first (from discovery or convention), then base model path
                var config = (configDir != null ? WhisperConfigReader.ReadConfig(configDir) : null)
                    ?? WhisperConfigReader.ReadConfig(baseModelPath);

                if (config != null)
                {
                    Trace.TraceInformation($"[OnnxTranscriberModel] Auto-detected config for fallback model '{_modelInfo.Id}' - " +
                        $"NumMelBins: {config.NumMelBins?.ToString(System.Globalization.CultureInfo.InvariantCulture) ?? "default"}, " +
                        $"HiddenSize: {config.HiddenSize?.ToString(System.Globalization.CultureInfo.InvariantCulture) ?? "default"}");
                    _modelInfo = _modelInfo.WithConfigOverrides(config);
                }
            }

            // Determine encoder/decoder paths using discovery result or fallback to legacy behavior
            string encoderPath;
            string decoderPath;
            string tokenizerPath;

            if (discovery != null)
            {
                // Use discovery result for accurate path resolution (handles subfolder structures)
                encoderPath = discovery.GetEncoderPath(baseModelPath)
                    ?? Path.Combine(discovery.GetOnnxDirectory(baseModelPath), _modelInfo!.EncoderFile);
                decoderPath = discovery.GetDecoderPath(baseModelPath)
                    ?? Path.Combine(discovery.GetOnnxDirectory(baseModelPath), _modelInfo!.DecoderFile);
                // Tokenizer files are typically in the base model directory
                tokenizerPath = baseModelPath;

                Trace.TraceInformation($"[OnnxTranscriberModel] Using discovery-based paths - Subfolder: {discovery.Subfolder ?? "(root)"}, " +
                    $"Encoder: {Path.GetFileName(encoderPath)}, Decoder: {Path.GetFileName(decoderPath)}");
            }
            else
            {
                // Fallback for local paths without discovery
                encoderPath = Path.Combine(baseModelPath, _modelInfo!.EncoderFile);
                decoderPath = Path.Combine(baseModelPath, _modelInfo.DecoderFile);
                tokenizerPath = baseModelPath;
            }

            // Load encoder with GPU provider verification
            if (!File.Exists(encoderPath))
            {
                throw new FileNotFoundException($"Encoder model not found: {encoderPath}");
            }

            _encoderSessionInfo = await OnnxSessionFactory.CreateWithInfoAsync(
                encoderPath,
                _options.Provider,
                ConfigureSessionOptions,
                cancellationToken: cancellationToken);
            _encoderSession = _encoderSessionInfo.Session;

            // Log GPU provider status
            Trace.TraceInformation($"[OnnxTranscriberModel] Encoder loaded - Requested: {_encoderSessionInfo.RequestedProvider}, " +
                $"Active providers: [{string.Join(", ", _encoderSessionInfo.ActiveProviders)}], GPU active: {_encoderSessionInfo.IsGpuActive}");

            if (_options.Provider != ExecutionProvider.Cpu && !_encoderSessionInfo.IsGpuActive)
            {
                Trace.TraceInformation("[OnnxTranscriberModel] WARNING: GPU provider was requested but only CPU is active. " +
                    "Check CUDA/DirectML installation and GPU availability.");
            }

            // Load decoder if available - use same provider as encoder
            if (File.Exists(decoderPath))
            {
                var decoderSessionInfo = await OnnxSessionFactory.CreateWithInfoAsync(
                    decoderPath,
                    _options.Provider,
                    ConfigureSessionOptions,
                    cancellationToken: cancellationToken);
                _decoderSession = decoderSessionInfo.Session;

                Trace.TraceInformation($"[OnnxTranscriberModel] Decoder loaded - Requested: {decoderSessionInfo.RequestedProvider}, " +
                    $"Active providers: [{string.Join(", ", decoderSessionInfo.ActiveProviders)}], GPU active: {decoderSessionInfo.IsGpuActive}");

                if (_options.Provider != ExecutionProvider.Cpu && !decoderSessionInfo.IsGpuActive)
                {
                    Trace.TraceInformation("[OnnxTranscriberModel] WARNING: Decoder GPU provider was requested but only CPU is active.");
                }
            }

            // Store model path and load tokenizer
            _modelPath = tokenizerPath;
            _tokenizer = await WhisperTokenizer.LoadAsync(tokenizerPath, cancellationToken);

            // Create decoder if decoder session is available
            if (_decoderSession != null)
            {
                _decoder = new WhisperDecoder(_decoderSession, _tokenizer);
            }

            _isInitialized = true;
        }
        catch
        {
            // Dispose partially-created sessions to prevent resource leaks
            _encoderSession?.Dispose();
            _encoderSession = null;
            _encoderSessionInfo = null;
            _decoderSession?.Dispose();
            _decoderSession = null;
            _decoder = null;
            throw;
        }
        finally
        {
            _lock.Release();
        }
    }

    private async Task<(string modelPath, ModelDiscoveryResult? discovery)> ResolveModelPathAsync(CancellationToken cancellationToken)
    {
        // If it's a local directory path, return it without discovery
        if (Directory.Exists(_modelInfo!.Id))
        {
            return (_modelInfo.Id, null);
        }

        // Check if parent directory exists (for file paths)
        var parentDir = Path.GetDirectoryName(_modelInfo.Id);
        if (parentDir != null && Directory.Exists(parentDir))
        {
            return (parentDir, null);
        }

        // Download from HuggingFace using discovery for complete file set
        var cacheDir = _options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        using var downloader = new HuggingFaceDownloader(cacheDir);

        // Build hardware-aware preferences with onnx subfolder for Whisper models
        var hwPrefs = ModelPreferences.ForCurrentHardware();
        var preferences = new ModelPreferences
        {
            PreferredSubfolder = "onnx",
            PreferLowMemory = hwPrefs.PreferLowMemory,
            QuantizationPriority = _options.QuantizationHint is { } hint
                ? ModelPreferences.ForQuantizationHint(hint).QuantizationPriority
                : hwPrefs.QuantizationPriority,
            PreferredProvider = _options.Provider != ExecutionProvider.Auto
                ? _options.Provider : hwPrefs.PreferredProvider,
            RequireMatchedQuantization = true
        };

        // Use discovery-based download to automatically find all model files
        // including external data files (*.onnx_data) for large models
        var (modelPath, discovery) = await downloader.DownloadWithDiscoveryAsync(
            _modelInfo.Id,
            preferences: preferences,
            cancellationToken: cancellationToken);

        return (modelPath, discovery);
    }

    private void ConfigureSessionOptions(SessionOptions options)
    {
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        options.EnableMemoryPattern = true;
        options.EnableCpuMemArena = true;
        options.LogSeverityLevel = (OrtLoggingLevel)(int)_options.LogLevel;

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
        _tokenizer?.Dispose();
        _lock.Dispose();
        return ValueTask.CompletedTask;
    }
}
