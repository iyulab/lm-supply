using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using LMSupply.Inference;
using LMSupply.Synthesizer.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.Synthesizer.Core;

/// <summary>
/// ONNX-based implementation of VITS/Piper TTS model.
/// </summary>
[System.Diagnostics.CodeAnalysis.Experimental(SynthesizerDiagnostics.ExperimentalId, UrlFormat = SynthesizerDiagnostics.ExperimentalUrl)]
internal sealed class OnnxSynthesizerModel : ISynthesizerModel
{
    private readonly SynthesizerOptions _options;
    private readonly SemaphoreSlim _lock = new(1, 1);

    // Owns the ONNX session and recovers from a crashing or hanging execution provider by moving
    // to the next one in the fallback chain (see RecoverableOnnxSession).
    private RecoverableOnnxSession? _session;
    private SynthesizerModelInfo? _modelInfo;
    private VoiceConfig? _voice;
    private bool _isInitialized;
    private bool _isDisposed;

    /// <inheritdoc />
    public string ModelId => _modelInfo?.Id ?? _options.ModelId;

    /// <inheritdoc />
    public bool IsGpuActive => _session?.IsGpuActive ?? false;

    /// <inheritdoc />
    public IReadOnlyList<string> ActiveProviders => _session?.ActiveProviders ?? Array.Empty<string>();

    /// <inheritdoc />
    public ExecutionProvider RequestedProvider => _options.Provider;

    /// <inheritdoc />
    public long? EstimatedMemoryBytes => _modelInfo?.SizeBytes * 2;

    public string? Voice => _modelInfo?.VoiceName;
    public int SampleRate => _voice?.SampleRate ?? _modelInfo?.SampleRate ?? 22050;

    public PhonemeScheme PhonemeScheme => _voice?.Scheme ?? throw new InvalidOperationException("The voice is not loaded.");

    // Reported by the download the first initialization makes (Local*.LoadAsync warms up at once).
    private readonly IProgress<DownloadProgress>? _downloadProgress;

    public OnnxSynthesizerModel(SynthesizerOptions options, IProgress<DownloadProgress>? downloadProgress = null)
    {
        _options = options.Clone();
        _downloadProgress = downloadProgress;
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
        ArgumentNullException.ThrowIfNull(text);
        await EnsureInitializedAsync(cancellationToken);

        var sw = Stopwatch.StartNew();
        var sentences = await PhonemizeAsync(text, cancellationToken);
        return await SynthesizeSentencesAsync(sentences, text, options, sw, cancellationToken);
    }

    public async Task<SynthesisResult> SynthesizePhonemesAsync(
        IReadOnlyList<string> sentences,
        SynthesizeOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(sentences);
        await EnsureInitializedAsync(cancellationToken);

        return await SynthesizeSentencesAsync(sentences, text: null, options, Stopwatch.StartNew(), cancellationToken);
    }

    public async Task SynthesizeToStreamAsync(
        string text,
        Stream outputStream,
        SynthesizeOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var result = await SynthesizeAsync(text, options, cancellationToken);
        await outputStream.WriteAsync(Format(result, options), cancellationToken);
    }

    public async Task SynthesizeToFileAsync(
        string text,
        string outputPath,
        SynthesizeOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        // Synthesize first: text that is refused leaves no file behind.
        var result = await SynthesizeAsync(text, options, cancellationToken);
        await File.WriteAllBytesAsync(outputPath, Format(result, options), cancellationToken);
    }

    private static byte[] Format(SynthesisResult result, SynthesizeOptions? options) =>
        (options?.OutputFormat ?? AudioFormat.Wav) switch
        {
            AudioFormat.RawPcm16 => result.ToPcm16Bytes(),
            AudioFormat.RawFloat32 => FloatToBytes(result.AudioSamples),
            _ => result.ToWavBytes()
        };

    public async IAsyncEnumerable<AudioChunk> SynthesizeStreamingAsync(
        string text,
        SynthesizeOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        await EnsureInitializedAsync(cancellationToken);

        var sentences = await PhonemizeAsync(text, cancellationToken);
        await foreach (var chunk in StreamSentencesAsync(sentences, options, cancellationToken))
            yield return chunk;
    }

    public async IAsyncEnumerable<AudioChunk> SynthesizePhonemesStreamingAsync(
        IReadOnlyList<string> sentences,
        SynthesizeOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(sentences);
        await EnsureInitializedAsync(cancellationToken);

        await foreach (var chunk in StreamSentencesAsync(sentences, options, cancellationToken))
            yield return chunk;
    }

    // Text becomes phonemes through the caller's phonemizer, or the built-in one for a voice whose phonemes are its
    // text. A voice with neither gets no audio: ids made up for its text would come out as noise.
    private async Task<IReadOnlyList<string>> PhonemizeAsync(string text, CancellationToken cancellationToken)
    {
        var scheme = _voice!.Scheme;
        var phonemizer = ResolvePhonemizer(_options.Phonemizer, scheme, _modelInfo!.AliasName);

        return await phonemizer.PhonemizeAsync(text, scheme, cancellationToken)
            ?? throw new InvalidOperationException($"{phonemizer.GetType().Name}.PhonemizeAsync returned null.");
    }

    internal static IPhonemizer ResolvePhonemizer(IPhonemizer? callers, PhonemeScheme scheme, string voiceName)
    {
        if (callers is not null)
            return callers;

        if (string.Equals(scheme.Type, VoiceConfig.TextType, StringComparison.OrdinalIgnoreCase))
            return CodepointPhonemizer.Instance;

        throw new NotSupportedException(
            $"The voice '{voiceName}' takes '{scheme.Type}' phonemes (phonemizer voice '{scheme.Voice ?? "unspecified"}', " +
            $"language '{scheme.Language ?? "unspecified"}'), and this package has no phonemizer that turns text into them. " +
            "Set SynthesizerOptions.Phonemizer to an IPhonemizer that produces them, or pass phonemes to SynthesizePhonemesAsync.");
    }

    private async Task<SynthesisResult> SynthesizeSentencesAsync(
        IReadOnlyList<string> sentences,
        string? text,
        SynthesizeOptions? options,
        Stopwatch sw,
        CancellationToken cancellationToken)
    {
        var (encoded, unknown) = Encode(sentences);
        var scales = _voice!.Scales(options);
        var speakerId = _voice.SpeakerId(options);

        var audio = new List<float[]>(encoded.Count);
        foreach (var sentence in encoded)
            audio.Add(await RunAsync(sentence.Ids, scales, speakerId, cancellationToken));

        var samples = new float[audio.Sum(a => a.Length)];
        var offset = 0;
        foreach (var part in audio)
        {
            part.CopyTo(samples, offset);
            offset += part.Length;
        }

        sw.Stop();
        return new SynthesisResult
        {
            AudioSamples = samples,
            SampleRate = SampleRate,
            InferenceTimeMs = sw.Elapsed.TotalMilliseconds,
            Text = text,
            Phonemes = [.. sentences],
            UnknownPhonemes = unknown
        };
    }

    private async IAsyncEnumerable<AudioChunk> StreamSentencesAsync(
        IReadOnlyList<string> sentences,
        SynthesizeOptions? options,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var (encoded, _) = Encode(sentences);
        var scales = _voice!.Scales(options);
        var speakerId = _voice.SpeakerId(options);

        for (var i = 0; i < encoded.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            yield return new AudioChunk
            {
                Samples = await RunAsync(encoded[i].Ids, scales, speakerId, cancellationToken),
                SampleRate = SampleRate,
                Index = i,
                IsFinal = i == encoded.Count - 1,
                UnknownPhonemes = encoded[i].Unknown
            };
        }
    }

    // Every sentence is encoded before the first one runs, so what the voice cannot say is known up front. A sentence
    // left with no phoneme the voice knows is dropped: its ids would be the start and end symbols alone.
    private (List<EncodedSentence> Encoded, string Unknown) Encode(IReadOnlyList<string> sentences)
    {
        var encoded = new List<EncodedSentence>(sentences.Count);
        var unknown = new List<Rune>();
        foreach (var sentence in sentences)
        {
            if (sentence is null)
                throw new ArgumentException("A sentence of phonemes is null.", nameof(sentences));

            var result = _voice!.Encoder.Encode(sentence);
            foreach (var phoneme in result.Unknown.EnumerateRunes())
            {
                if (!unknown.Contains(phoneme))
                    unknown.Add(phoneme);
            }

            if (result.PhonemeCount > 0)
                encoded.Add(result);
        }

        if (unknown.Count > 0)
        {
            Trace.TraceWarning(
                $"[OnnxSynthesizerModel] {ModelId}: skipped {unknown.Count} phoneme(s) the voice has no id for: " +
                string.Join(" ", unknown.Select(r => $"U+{r.Value:X4}")));
        }

        return (encoded, string.Concat(unknown));
    }

    private async Task<float[]> RunAsync(long[] phonemeIds, float[] scales, long? speakerId, CancellationToken cancellationToken)
    {
        await _lock.WaitAsync(cancellationToken);
        try
        {
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", new DenseTensor<long>(phonemeIds, [1, phonemeIds.Length])),
                NamedOnnxValue.CreateFromTensor("input_lengths", new DenseTensor<long>(new long[] { phonemeIds.Length }, [1])),
                NamedOnnxValue.CreateFromTensor("scales", new DenseTensor<float>(scales, [scales.Length]))
            };

            // The graph of a multi-speaker voice has a speaker input; a single-speaker one does not.
            if (speakerId is { } sid)
                inputs.Add(NamedOnnxValue.CreateFromTensor("sid", new DenseTensor<long>(new[] { sid }, [1])));

            // Bounded run: if the native call hangs (e.g. a cold GPU kernel init) or the
            // provider crashes, the session moves to the next provider and the run is retried once.
            return await _session!.RunWithRecoveryAsync((session, runOptions) =>
            {
                using var results = session.Run(inputs, session.OutputNames, runOptions);
                return results[0].AsTensor<float>().ToArray();
            }, cancellationToken: cancellationToken);
        }
        finally
        {
            _lock.Release();
        }
    }

    private async Task EnsureInitializedAsync(CancellationToken cancellationToken)
    {
        if (_isInitialized) return;

        await _lock.WaitAsync(cancellationToken);
        try
        {
            if (_isInitialized) return;

            // Resolve model info
            _modelInfo = SynthesizerModelRegistry.Default.Resolve(_options.ModelId);

            // Download model if needed
            var modelPath = await ResolveModelPathAsync(cancellationToken);

            // The config is the voice's contract: its phoneme ids, inference defaults, sample rate and speaker count.
            // A voice without one cannot be run correctly, so it is not run at all.
            var configPath = Path.Combine(modelPath, _modelInfo!.ConfigFile);
            if (!File.Exists(configPath))
            {
                throw new FileNotFoundException($"Voice config not found: {configPath}");
            }

            _voice = VoiceConfig.Parse(await File.ReadAllTextAsync(configPath, cancellationToken), _modelInfo);
            _modelInfo = _voice.Reconcile(_modelInfo);

            // Load model
            var modelFilePath = Path.Combine(modelPath, _modelInfo.ModelFile);
            if (!File.Exists(modelFilePath))
            {
                throw new FileNotFoundException($"Model file not found: {modelFilePath}");
            }

            var result = await OnnxSessionFactory.CreateWithInfoAsync(
                modelFilePath,
                _options.Provider,
                ConfigureSessionOptions,
                _downloadProgress,
                cancellationToken);

            _session = RecoverableOnnxSession.FromResult(
                result, modelFilePath, ConfigureSessionOptions, logPrefix: "[OnnxSynthesizerModel]");

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
        using var downloader = new HuggingFaceDownloader(cacheDir, localFilesOnly: _options.DisableAutoDownload);

        var modelPath = await downloader.DownloadModelAsync(
            _modelInfo.Id,
            files: [_modelInfo.ModelFile, _modelInfo.ConfigFile],
            subfolder: _modelInfo.VoiceName,
            progress: _downloadProgress,
            cancellationToken: cancellationToken);

        return modelPath;
    }

    private void ConfigureSessionOptions(SessionOptions options)
    {
        options.ApplyCommonOptions(_options);
    }

    private static byte[] FloatToBytes(float[] samples)
    {
        var bytes = new byte[samples.Length * 4];
        Buffer.BlockCopy(samples, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    public ValueTask DisposeAsync()
    {
        if (_isDisposed) return ValueTask.CompletedTask;
        _isDisposed = true;

        _session?.Dispose();
        _lock.Dispose();
        return ValueTask.CompletedTask;
    }
}
