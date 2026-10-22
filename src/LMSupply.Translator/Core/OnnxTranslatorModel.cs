using System.Diagnostics;
using LMSupply.Core.Download;
using LMSupply.Download;
using LMSupply.Inference;
using LMSupply.Translator.Decoding;
using LMSupply.Translator.Models;
using LMSupply.Translator.Tokenization;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.Translator.Core;

/// <summary>
/// ONNX Runtime-based machine translation implementation.
/// Supports OPUS-MT (MarianMT) encoder-decoder architecture.
/// </summary>
internal sealed class OnnxTranslatorModel : ITranslatorModel
{

    private readonly TranslatorOptions _options;
    private readonly TranslatorModelInfo _modelInfo;
    private readonly SemaphoreSlim _sessionLock = new(1, 1);

    // Encoder and decoder each own a RecoverableOnnxSession and share one provider blacklist, so a
    // provider that crashes or hangs on one half of the model is left by the other half before its
    // next run (see RecoverableOnnxSession).
    private readonly ProviderBlacklist _providerBlacklist = new();
    private RecoverableOnnxSession? _encoderSession;
    private RecoverableOnnxSession? _decoderSession;
    private TranslatorTokenizer? _tokenizer;
    private bool _isInitialized;
    private bool _disposed;

    // Resolved file paths (may be auto-discovered or explicit)
    private string? _resolvedEncoderFile;
    private string? _resolvedDecoderFile;

    /// <inheritdoc />
    public string ModelId => _modelInfo.Id;

    /// <inheritdoc />
    public bool IsGpuActive => _encoderSession?.IsGpuActive ?? false;

    /// <inheritdoc />
    public IReadOnlyList<string> ActiveProviders => _encoderSession?.ActiveProviders ?? Array.Empty<string>();

    /// <inheritdoc />
    public ExecutionProvider RequestedProvider => _options.Provider;

    /// <inheritdoc />
    public long? EstimatedMemoryBytes => _modelInfo.SizeBytes * 2;

    /// <summary>
    /// Gets the source language code.
    /// </summary>
    public string SourceLanguage => _modelInfo.SourceLanguage;

    /// <summary>
    /// Gets the target language code.
    /// </summary>
    public string TargetLanguage => _modelInfo.TargetLanguage;

    // Reported by the download the first initialization makes (Local*.LoadAsync warms up at once).
    private readonly IProgress<DownloadProgress>? _downloadProgress;

    public OnnxTranslatorModel(TranslatorOptions options, IProgress<DownloadProgress>? downloadProgress = null)
    {
        _options = options.Clone();
        _downloadProgress = downloadProgress;
        _modelInfo = TranslatorModelRegistry.Default.Resolve(options.ModelId);
    }

    public async Task WarmupAsync(CancellationToken cancellationToken = default)
    {
        await EnsureInitializedAsync(cancellationToken);
    }

    public TranslatorModelInfo? GetModelInfo() => _modelInfo;

    public async Task<TranslationResult> TranslateAsync(
        string text,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(text);

        await EnsureInitializedAsync(cancellationToken);

        var sw = Stopwatch.StartNew();

        // Tokenize input
        var inputIds = Tokenize(text);

        // Create attention mask (1 for real tokens, 0 for padding)
        var attentionMask = inputIds.Select(id => id != _tokenizer!.PadTokenId ? 1L : 0L).ToArray();

        // Run encoder
        var encoderOutput = await RunEncoderAsync(inputIds, attentionMask, cancellationToken);

        // Run decoder with selected decoding strategy
        long[] outputIds;
        if (_options.UseGreedyDecoding)
        {
            outputIds = await RunDecoderGreedyAsync(encoderOutput, attentionMask, cancellationToken);
        }
        else
        {
            outputIds = await RunDecoderBeamSearchAsync(encoderOutput, attentionMask, cancellationToken);
        }

        // Detokenize output
        var translatedText = Detokenize(outputIds);

        sw.Stop();

        return new TranslationResult
        {
            SourceText = text,
            TranslatedText = translatedText,
            SourceLanguage = _modelInfo.SourceLanguage,
            TargetLanguage = _modelInfo.TargetLanguage,
            InferenceTimeMs = sw.Elapsed.TotalMilliseconds
        };
    }

    public async Task<IReadOnlyList<TranslationResult>> TranslateBatchAsync(
        IEnumerable<string> texts,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(texts);

        var results = new List<TranslationResult>();

        foreach (var text in texts)
        {
            var result = await TranslateAsync(text, cancellationToken);
            results.Add(result);
        }

        return results;
    }

    private long[] Tokenize(string text)
    {
        // Use SentencePiece tokenizer for proper subword tokenization
        return _tokenizer!.EncodeSource(text, addSpecialTokens: true);
    }

    private string Detokenize(long[] tokenIds)
    {
        // Use SentencePiece tokenizer for proper detokenization
        return _tokenizer!.DecodeTarget(tokenIds, skipSpecialTokens: true);
    }

    private async Task<DenseTensor<float>> RunEncoderAsync(
        long[] inputIds,
        long[] attentionMask,
        CancellationToken cancellationToken)
    {
        var seqLen = inputIds.Length;

        // Create input tensors
        var inputIdsTensor = new DenseTensor<long>(inputIds, [1, seqLen]);
        var attentionMaskTensor = new DenseTensor<long>(attentionMask, [1, seqLen]);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor)
        };

        await _sessionLock.WaitAsync(cancellationToken);
        try
        {
            // Bounded run: if the native call hangs (e.g. a cold GPU kernel init) or the
            // provider crashes, the session moves to the next provider and the run is retried once.
            return await _encoderSession!.RunWithRecoveryAsync((session, runOptions) =>
            {
                using var outputs = session.Run(inputs, session.OutputNames, runOptions);
                var lastHiddenState = outputs[0].AsTensor<float>();

                // Copy out of the native buffer since the outputs are disposed with this delegate
                return new DenseTensor<float>(lastHiddenState.ToArray(), lastHiddenState.Dimensions);
            }, cancellationToken: cancellationToken);
        }
        finally
        {
            _sessionLock.Release();
        }
    }

    private async Task<long[]> RunDecoderGreedyAsync(
        DenseTensor<float> encoderOutput,
        long[] encoderAttentionMask,
        CancellationToken cancellationToken)
    {
        // MarianMT uses pad_token_id as decoder_start_token_id
        var outputIds = new List<long>(_tokenizer!.GetDecoderStartIds());
        var maxLength = Math.Min(_options.MaxLength, _modelInfo.MaxLength);
        var eosTokenId = _tokenizer.EosTokenId;

        await _sessionLock.WaitAsync(cancellationToken);
        try
        {
            // No KV caching: a merged export still needs use_cache_branch=false and a stand-in for every past input.
            // Metadata is identical on every provider, so reading it once is safe across recoveries.
            var noCacheInputs = new NoCacheDecoderInputs(_decoderSession!.Session.InputMetadata);

            for (int step = 0; step < maxLength; step++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var decoderInputIds = new DenseTensor<long>(outputIds.ToArray(), [1, outputIds.Count]);
                var encoderAttention = new DenseTensor<long>(encoderAttentionMask, [1, encoderAttentionMask.Length]);

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("input_ids", decoderInputIds),
                    NamedOnnxValue.CreateFromTensor("encoder_attention_mask", encoderAttention),
                    NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderOutput)
                };

                noCacheInputs.AddTo(inputs);

                var lastPosition = outputIds.Count - 1;

                // One bounded decode step; a provider crash or hang inside it moves the session to
                // the next provider and retries this step (decoder state is on the managed side).
                var bestToken = await _decoderSession.RunWithRecoveryAsync((session, runOptions) =>
                {
                    using var outputs = session.Run(inputs, [session.OutputNames[0]], runOptions);
                    var logits = outputs[0].AsTensor<float>();
                    var vocabSize = (int)logits.Dimensions[2];

                    // Greedy decoding: select token with highest probability at the last position
                    float maxLogit = float.MinValue;
                    long best = eosTokenId;
                    for (int v = 0; v < vocabSize; v++)
                    {
                        var logit = logits[0, lastPosition, v];
                        if (logit > maxLogit)
                        {
                            maxLogit = logit;
                            best = v;
                        }
                    }
                    return best;
                }, cancellationToken: cancellationToken);

                outputIds.Add(bestToken);

                // Stop if EOS token generated
                if (bestToken == eosTokenId)
                    break;
            }
        }
        finally
        {
            _sessionLock.Release();
        }

        return outputIds.ToArray();
    }

    private async Task<long[]> RunDecoderBeamSearchAsync(
        DenseTensor<float> encoderOutput,
        long[] encoderAttentionMask,
        CancellationToken cancellationToken)
    {
        var decoder = new BeamSearchDecoder(
            beamSize: _options.BeamWidth,
            maxLength: Math.Min(_options.MaxLength, _modelInfo.MaxLength),
            eosTokenId: _tokenizer!.EosTokenId,
            padTokenId: _tokenizer.PadTokenId,
            lengthPenalty: _options.LengthPenalty,
            repetitionPenalty: _options.RepetitionPenalty);

        var startTokenIds = _tokenizer.GetDecoderStartIds();

        await _sessionLock.WaitAsync(cancellationToken);
        try
        {
            return await decoder.DecodeAsync(
                encoderOutput,
                encoderAttentionMask,
                startTokenIds,
                _decoderSession!,
                cancellationToken);
        }
        finally
        {
            _sessionLock.Release();
        }
    }

    private async Task EnsureInitializedAsync(CancellationToken cancellationToken)
    {
        if (_isInitialized)
            return;

        await _sessionLock.WaitAsync(cancellationToken);
        try
        {
            if (_isInitialized)
                return;

            var modelDir = await ResolveModelPathAsync(cancellationToken);

            // Use resolved file paths (set during ResolveModelPathAsync)
            var encoderFile = _resolvedEncoderFile ?? _modelInfo.EncoderFile ?? "encoder_model.onnx";
            var decoderFile = _resolvedDecoderFile ?? _modelInfo.DecoderFile ?? "decoder_model.onnx";

            // Load encoder
            var encoderPath = Path.Combine(modelDir, encoderFile);
            var encoderResult = await OnnxSessionFactory.CreateWithInfoAsync(
                encoderPath,
                _options.Provider,
                ConfigureSessionOptions,
                _downloadProgress,
                cancellationToken);

            _encoderSession = RecoverableOnnxSession.FromResult(
                encoderResult, encoderPath, ConfigureSessionOptions,
                logPrefix: "[OnnxTranslatorModel:encoder]", blacklist: _providerBlacklist);

            // Load decoder
            var decoderPath = Path.Combine(modelDir, decoderFile);
            var decoderResult = await OnnxSessionFactory.CreateWithInfoAsync(
                decoderPath,
                _options.Provider,
                ConfigureSessionOptions,
                _downloadProgress,
                cancellationToken);
            _decoderSession = RecoverableOnnxSession.FromResult(
                decoderResult, decoderPath, ConfigureSessionOptions,
                logPrefix: "[OnnxTranslatorModel:decoder]", blacklist: _providerBlacklist);

            // Initialize SentencePiece tokenizer
            _tokenizer = TranslatorTokenizer.Create(modelDir);

            _isInitialized = true;
        }
        finally
        {
            _sessionLock.Release();
        }
    }

    private async Task<string> ResolveModelPathAsync(CancellationToken cancellationToken)
    {
        var (modelDir, encoderFile, decoderFile) = await DownloadAsync(_modelInfo, _options, _downloadProgress, cancellationToken);
        _resolvedEncoderFile = encoderFile;
        _resolvedDecoderFile = decoderFile;
        return modelDir;
    }

    /// <summary>
    /// The directory a model on local disk is read from (the id itself, or the directory that holds it), or null for a
    /// repository the load downloads.
    /// </summary>
    internal static string? LocalDirectory(TranslatorModelInfo modelInfo)
    {
        if (Directory.Exists(modelInfo.Id))
            return modelInfo.Id;

        var parentDir = Path.GetDirectoryName(modelInfo.Id);
        return parentDir != null && Directory.Exists(parentDir) ? parentDir : null;
    }

    /// <summary>
    /// The download a load makes — or, with <see cref="TranslatorOptions.DisableAutoDownload"/>, the cache read — and the
    /// encoder and decoder files it opens, relative to the directory. Shared by the load and
    /// <see cref="LocalTranslator.DownloadModelAsync"/>, so both take the same files.
    /// </summary>
    internal static async Task<(string ModelDir, string EncoderFile, string DecoderFile)> DownloadAsync(
        TranslatorModelInfo modelInfo, TranslatorOptions options, IProgress<DownloadProgress>? progress, CancellationToken cancellationToken)
    {
        if (LocalDirectory(modelInfo) is { } localDir)
            return (localDir, modelInfo.EncoderFile ?? "encoder_model.onnx", modelInfo.DecoderFile ?? "decoder_model.onnx");

        using var downloader = new HuggingFaceDownloader(CacheDirectory(options), localFilesOnly: options.DisableAutoDownload);

        if (modelInfo.UseAutoDiscovery)
        {
            var (modelDir, discovery) = await downloader.DownloadWithDiscoveryAsync(
                modelInfo.Id,
                preferences: DiscoveryPreferences(modelInfo, options),
                progress: progress,
                cancellationToken: cancellationToken);

            // Discovered file paths keep their relative structure, for models in a subfolder
            var encoderFile = discovery.PrimaryEncoderFile is not null
                ? discovery.PrimaryEncoderFile.Replace('/', Path.DirectorySeparatorChar)
                : throw new InvalidOperationException($"No encoder model found in repository '{modelInfo.Id}'");

            var decoderFile = discovery.PrimaryDecoderFile is not null
                ? discovery.PrimaryDecoderFile.Replace('/', Path.DirectorySeparatorChar)
                : throw new InvalidOperationException($"No decoder model found in repository '{modelInfo.Id}'");

            return (modelDir, encoderFile, decoderFile);
        }

        var explicitDir = await downloader.DownloadModelAsync(
            modelInfo.Id,
            files: RequiredFiles(modelInfo),
            progress: progress,
            cancellationToken: cancellationToken);

        return (explicitDir, modelInfo.EncoderFile ?? "encoder_model.onnx", modelInfo.DecoderFile ?? "decoder_model.onnx");
    }

    /// <summary>
    /// The files <see cref="DownloadAsync"/> would fetch for the same model and options, with their listed lengths; null for
    /// a model on local disk. Chosen by the same discovery or file list, so the plan and the download cannot disagree.
    /// Downloads nothing.
    /// </summary>
    internal static async Task<DownloadPlan?> PlanAsync(
        TranslatorModelInfo modelInfo, TranslatorOptions options, CancellationToken cancellationToken)
    {
        if (LocalDirectory(modelInfo) is not null)
            return null;

        using var downloader = new HuggingFaceDownloader(CacheDirectory(options), localFilesOnly: options.DisableAutoDownload);
        return modelInfo.UseAutoDiscovery
            ? await downloader.PlanWithDiscoveryAsync(
                modelInfo.Id, preferences: DiscoveryPreferences(modelInfo, options), cancellationToken: cancellationToken)
            : await downloader.PlanModelAsync(modelInfo.Id, files: RequiredFiles(modelInfo), cancellationToken: cancellationToken);
    }

    internal static string CacheDirectory(TranslatorOptions options) =>
        options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();

    private static ModelPreferences DiscoveryPreferences(TranslatorModelInfo modelInfo, TranslatorOptions options)
    {
        var hwPrefs = ModelPreferences.ForProvider(options.Provider);
        return new ModelPreferences
        {
            PreferredSubfolder = modelInfo.Subfolder,
            DecoderVariantPriority = [modelInfo.PreferredDecoderVariant, DecoderVariant.Standard, DecoderVariant.Merged],
            QuantizationPriority = options.QuantizationHint is { } hint
                ? ModelPreferences.ForQuantizationHint(hint).QuantizationPriority
                : hwPrefs.QuantizationPriority,
            PreferredProvider = options.Provider != ExecutionProvider.Auto
                ? options.Provider : hwPrefs.PreferredProvider
        };
    }

    // The files of a model that names them rather than being discovered
    private static string[] RequiredFiles(TranslatorModelInfo modelInfo) =>
    [
        modelInfo.EncoderFile ?? "encoder_model.onnx",
        modelInfo.DecoderFile ?? "decoder_model.onnx",
        "config.json",
        "vocab.json",
        "tokenizer.json",
        modelInfo.TokenizerFile
    ];

    private void ConfigureSessionOptions(SessionOptions options)
    {
        options.ApplyCommonOptions(_options);

        options.EnableMemoryPattern = true;
        options.EnableCpuMemArena = true;
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        await _sessionLock.WaitAsync();
        try
        {
            _encoderSession?.Dispose();
            _decoderSession?.Dispose();
            _tokenizer?.Dispose();
        }
        finally
        {
            _sessionLock.Release();
            _sessionLock.Dispose();
        }

        _disposed = true;
    }
}
