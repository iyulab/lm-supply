using System.Diagnostics;
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

    private InferenceSession? _encoderSession;
    private InferenceSession? _decoderSession;
    private TranslatorTokenizer? _tokenizer;
    private bool _isInitialized;
    private bool _disposed;

    /// <inheritdoc />
    public string ModelId => _modelInfo.Id;

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
        await _sessionLock.WaitAsync(cancellationToken);
        try
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

            using var outputs = _encoderSession!.Run(inputs);
            var lastHiddenState = outputs.First().AsTensor<float>();

            // Clone the tensor since we're disposing the outputs
            var dims = lastHiddenState.Dimensions.ToArray();
            var result = new DenseTensor<float>(dims);

            // Copy data element by element
            var totalElements = dims.Aggregate(1, (a, b) => a * b);
            for (int i = 0; i < totalElements; i++)
            {
                result.SetValue(i, lastHiddenState.GetValue(i));
            }

            return result;
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

                using var outputs = _decoderSession!.Run(inputs);
                var logits = outputs.First().AsTensor<float>();

                // Get the last token's logits
                var lastPosition = outputIds.Count - 1;
                var vocabSize = (int)logits.Dimensions[2];

                // Greedy decoding: select token with highest probability
                float maxLogit = float.MinValue;
                long bestToken = eosTokenId;

                for (int v = 0; v < vocabSize; v++)
                {
                    var logit = logits[0, lastPosition, v];
                    if (logit > maxLogit)
                    {
                        maxLogit = logit;
                        bestToken = v;
                    }
                }

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

            // Load encoder
            var encoderPath = Path.Combine(modelDir, _modelInfo.EncoderFile);
            _encoderSession = await OnnxSessionFactory.CreateAsync(
                encoderPath,
                _options.Provider,
                ConfigureSessionOptions,
                cancellationToken: cancellationToken);

            // Load decoder
            var decoderPath = Path.Combine(modelDir, _modelInfo.DecoderFile);
            _decoderSession = await OnnxSessionFactory.CreateAsync(
                decoderPath,
                _options.Provider,
                ConfigureSessionOptions,
                cancellationToken: cancellationToken);

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
            return _modelInfo.Id;
        }

        var parentDir = Path.GetDirectoryName(_modelInfo.Id);
        if (parentDir != null && Directory.Exists(parentDir))
        {
            return parentDir;
        }

        // Download from HuggingFace
        using var downloader = new HuggingFaceDownloader(_options.CacheDirectory);

        var modelDir = await downloader.DownloadModelAsync(
            _modelInfo.Id,
            files:
            [
                _modelInfo.EncoderFile,
                _modelInfo.DecoderFile,
                "config.json",
                "vocab.json",
                "tokenizer.json",
                _modelInfo.TokenizerFile
            ],
            cancellationToken: cancellationToken);

        return modelDir;
    }

    private void ConfigureSessionOptions(SessionOptions options)
    {
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
