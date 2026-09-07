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
    private static readonly bool[] s_falseArray = [false];
    private static readonly int[] s_oneDimension = [1];

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

    public OnnxTranslatorModel(TranslatorOptions options)
    {
        _options = options.Clone();
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
            // Bounded run: if the native call hangs (e.g. a cold DirectML kernel init) or the
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
            // use_cache_branch is required by merged decoder exports; always false (no KV-caching).
            // Metadata is identical on every provider, so reading it once is safe across recoveries.
            var needsCacheBranch = _decoderSession!.Session.InputMetadata.ContainsKey("use_cache_branch");

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

                if (needsCacheBranch)
                {
                    var useCacheBranch = new DenseTensor<bool>(s_falseArray, s_oneDimension);
                    inputs.Add(NamedOnnxValue.CreateFromTensor("use_cache_branch", useCacheBranch));
                }

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
                cancellationToken: cancellationToken);

            _encoderSession = RecoverableOnnxSession.FromResult(
                encoderResult, encoderPath, ConfigureSessionOptions,
                logPrefix: "[OnnxTranslatorModel:encoder]", blacklist: _providerBlacklist);

            // Load decoder
            var decoderPath = Path.Combine(modelDir, decoderFile);
            var decoderResult = await OnnxSessionFactory.CreateWithInfoAsync(
                decoderPath,
                _options.Provider,
                ConfigureSessionOptions,
                cancellationToken: cancellationToken);
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
        // If it's a local path, return directory
        if (Directory.Exists(_modelInfo.Id))
        {
            // For local paths, use explicit file names if provided
            _resolvedEncoderFile = _modelInfo.EncoderFile;
            _resolvedDecoderFile = _modelInfo.DecoderFile;
            return _modelInfo.Id;
        }

        var parentDir = Path.GetDirectoryName(_modelInfo.Id);
        if (parentDir != null && Directory.Exists(parentDir))
        {
            _resolvedEncoderFile = _modelInfo.EncoderFile;
            _resolvedDecoderFile = _modelInfo.DecoderFile;
            return parentDir;
        }

        // Download from HuggingFace
        using var downloader = new HuggingFaceDownloader(_options.CacheDirectory);

        if (_modelInfo.UseAutoDiscovery)
        {
            // Use auto-discovery for ONNX files
            var hwPrefs = ModelPreferences.ForCurrentHardware();
            var preferences = new ModelPreferences
            {
                PreferredSubfolder = _modelInfo.Subfolder,
                DecoderVariantPriority = [_modelInfo.PreferredDecoderVariant, DecoderVariant.Standard, DecoderVariant.Merged],
                QuantizationPriority = _options.QuantizationHint is { } hint
                    ? ModelPreferences.ForQuantizationHint(hint).QuantizationPriority
                    : hwPrefs.QuantizationPriority,
                PreferredProvider = _options.Provider != ExecutionProvider.Auto
                    ? _options.Provider : hwPrefs.PreferredProvider
            };

            var (modelDir, discovery) = await downloader.DownloadWithDiscoveryAsync(
                _modelInfo.Id,
                preferences: preferences,
                cancellationToken: cancellationToken);

            // Store discovered file paths (preserve relative path structure for subfolder support)
            _resolvedEncoderFile = discovery.PrimaryEncoderFile is not null
                ? discovery.PrimaryEncoderFile.Replace('/', Path.DirectorySeparatorChar)
                : throw new InvalidOperationException($"No encoder model found in repository '{_modelInfo.Id}'");

            _resolvedDecoderFile = discovery.PrimaryDecoderFile is not null
                ? discovery.PrimaryDecoderFile.Replace('/', Path.DirectorySeparatorChar)
                : throw new InvalidOperationException($"No decoder model found in repository '{_modelInfo.Id}'");

            return modelDir;
        }
        else
        {
            // Use explicit file list (legacy behavior)
            var encoderFile = _modelInfo.EncoderFile ?? "encoder_model.onnx";
            var decoderFile = _modelInfo.DecoderFile ?? "decoder_model.onnx";

            var modelDir = await downloader.DownloadModelAsync(
                _modelInfo.Id,
                files:
                [
                    encoderFile,
                    decoderFile,
                    "config.json",
                    "vocab.json",
                    "tokenizer.json",
                    _modelInfo.TokenizerFile
                ],
                cancellationToken: cancellationToken);

            _resolvedEncoderFile = encoderFile;
            _resolvedDecoderFile = decoderFile;

            return modelDir;
        }
    }

    private void ConfigureSessionOptions(SessionOptions options)
    {
        options.LogSeverityLevel = (OrtLoggingLevel)(int)_options.LogLevel;

        if (_options.ThreadCount.HasValue)
        {
            options.IntraOpNumThreads = _options.ThreadCount.Value;
            options.InterOpNumThreads = _options.ThreadCount.Value;
        }

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
