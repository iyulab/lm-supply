using System.Diagnostics;
using System.Runtime.CompilerServices;
using LMSupply.Generator.Abstractions;
using LMSupply.Generator.Models;

namespace LMSupply.Generator;

/// <summary>
/// Basic implementation of speculative decoding using draft and target models.
/// </summary>
/// <remarks>
/// This implementation uses a smaller, faster draft model to generate candidate
/// tokens which are then verified by a larger, more accurate target model.
/// The speedup comes from batched verification of multiple draft tokens at once.
/// </remarks>
public sealed class SpeculativeDecoder : ISpeculativeDecoder, IDisposable
{
    private readonly IGeneratorModel _draftModel;
    private readonly IGeneratorModel _targetModel;
    private readonly SpeculativeDecodingOptions _options;
    private SpeculativeStats _lastStats = CreateEmptyStats();
    private bool _disposed;

    /// <summary>
    /// Creates a new speculative decoder with draft and target models.
    /// </summary>
    /// <param name="draftModel">The smaller, faster draft model.</param>
    /// <param name="targetModel">The larger, more accurate target model.</param>
    /// <param name="options">Speculative decoding options.</param>
    public SpeculativeDecoder(
        IGeneratorModel draftModel,
        IGeneratorModel targetModel,
        SpeculativeDecodingOptions? options = null)
    {
        _draftModel = draftModel ?? throw new ArgumentNullException(nameof(draftModel));
        _targetModel = targetModel ?? throw new ArgumentNullException(nameof(targetModel));
        _options = options ?? new SpeculativeDecodingOptions();
        SpeculationLength = _options.SpeculationLength;
    }

    /// <inheritdoc />
    public string DraftModelId => _draftModel.ModelId;

    /// <inheritdoc />
    public string TargetModelId => _targetModel.ModelId;

    /// <inheritdoc />
    public int SpeculationLength { get; set; }

    /// <inheritdoc />
    public async IAsyncEnumerable<SpeculativeToken> GenerateAsync(
        string prompt,
        GenerationOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        var sw = Stopwatch.StartNew();
        var totalTokens = 0;
        var draftTokens = 0;
        var acceptedTokens = 0;
        var targetTokens = 0;

        var currentPrompt = prompt;
        var generatedText = string.Empty;
        var maxTokens = options?.MaxTokens ?? 256;

        // Draft options (lower temperature for more predictable outputs)
        var draftOptions = CreateDraftOptions(options);

        while (totalTokens < maxTokens && !cancellationToken.IsCancellationRequested)
        {
            // Step 1: Generate speculative tokens with draft model
            var draftCandidates = new List<string>();
            var speculationOptions = CloneWithMaxTokens(draftOptions, SpeculationLength);
            await foreach (var token in _draftModel.GenerateAsync(
                currentPrompt + generatedText,
                speculationOptions,
                cancellationToken))
            {
                draftCandidates.Add(token);
                if (draftCandidates.Count >= SpeculationLength) break;
            }

            if (draftCandidates.Count == 0) break;
            draftTokens += draftCandidates.Count;

            // Step 2: Verify with target model
            var (verified, rejected) = await VerifyTokensAsync(
                currentPrompt + generatedText,
                draftCandidates,
                options,
                cancellationToken);

            // Yield verified tokens
            foreach (var token in verified)
            {
                yield return new SpeculativeToken(token, WasSpeculated: true, WasAccepted: true);
                generatedText += token;
                totalTokens++;
                acceptedTokens++;

                if (totalTokens >= maxTokens) break;
            }

            // If some tokens were rejected, generate correction from target
            if (rejected && totalTokens < maxTokens)
            {
                var targetToken = await GenerateTargetTokenAsync(
                    currentPrompt + generatedText,
                    options,
                    cancellationToken);

                if (!string.IsNullOrEmpty(targetToken))
                {
                    yield return new SpeculativeToken(targetToken, WasSpeculated: false, WasAccepted: true);
                    generatedText += targetToken;
                    totalTokens++;
                    targetTokens++;
                }
            }

            // Adaptive speculation length adjustment
            if (_options.AdaptiveSpeculation && draftTokens > 0)
            {
                var currentRate = (double)acceptedTokens / draftTokens;
                if (currentRate < _options.MinAcceptanceRate && SpeculationLength > 1)
                {
                    SpeculationLength--;
                }
                else if (currentRate > 0.8 && SpeculationLength < 10)
                {
                    SpeculationLength++;
                }
            }

            // Check for end of generation (simple heuristic)
            if (generatedText.EndsWith('\n') && generatedText.TrimEnd().EndsWith('.'))
            {
                break;
            }
        }

        sw.Stop();
        _lastStats = new SpeculativeStats
        {
            TotalTokens = totalTokens,
            DraftTokens = draftTokens,
            AcceptedTokens = acceptedTokens,
            TargetTokens = targetTokens,
            ElapsedMilliseconds = sw.ElapsedMilliseconds
        };
    }

    /// <inheritdoc />
    public async Task<SpeculativeResult> GenerateCompleteAsync(
        string prompt,
        GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        var tokens = new List<string>();
        await foreach (var token in GenerateAsync(prompt, options, cancellationToken))
        {
            tokens.Add(token.Token);
        }

        var text = string.Join(string.Empty, tokens);
        return new SpeculativeResult(text, _lastStats);
    }

    /// <inheritdoc />
    public SpeculativeStats GetLastStats() => _lastStats;

    private GenerationOptions CreateDraftOptions(GenerationOptions? baseOptions)
    {
        var opts = baseOptions ?? new GenerationOptions();

        // Use lower temperature for draft model for more predictable tokens
        var draftTemp = _options.DraftTemperature ?? Math.Max(0.1f, opts.Temperature - 0.2f);

        return new GenerationOptions
        {
            MaxTokens = opts.MaxTokens,
            Temperature = draftTemp,
            TopP = opts.TopP,
            TopK = opts.TopK,
            RepetitionPenalty = opts.RepetitionPenalty,
            StopSequences = opts.StopSequences,
            IncludePromptInOutput = opts.IncludePromptInOutput,
            DoSample = opts.DoSample,
            NumBeams = opts.NumBeams,
            PastPresentShareBuffer = opts.PastPresentShareBuffer,
            MaxNewTokens = opts.MaxNewTokens
        };
    }

    private static GenerationOptions CloneWithMaxTokens(GenerationOptions source, int maxTokens)
    {
        return new GenerationOptions
        {
            MaxTokens = maxTokens,
            Temperature = source.Temperature,
            TopP = source.TopP,
            TopK = source.TopK,
            RepetitionPenalty = source.RepetitionPenalty,
            StopSequences = source.StopSequences,
            IncludePromptInOutput = source.IncludePromptInOutput,
            DoSample = source.DoSample,
            NumBeams = source.NumBeams,
            PastPresentShareBuffer = source.PastPresentShareBuffer,
            MaxNewTokens = source.MaxNewTokens
        };
    }

    private async Task<(List<string> verified, bool rejected)> VerifyTokensAsync(
        string context,
        List<string> candidates,
        GenerationOptions? options,
        CancellationToken cancellationToken)
    {
        // Simplified verification: generate from target and compare
        // In a real implementation, this would use probability comparison
        var verified = new List<string>();
        var rejected = false;

        // Generate same number of tokens from target model
        var targetTokens = new List<string>();
        var verifyOptions = CloneWithMaxTokens(options ?? new GenerationOptions(), candidates.Count);
        await foreach (var token in _targetModel.GenerateAsync(
            context,
            verifyOptions,
            cancellationToken))
        {
            targetTokens.Add(token);
            if (targetTokens.Count >= candidates.Count) break;
        }

        // Compare tokens (simplified: exact match)
        for (int i = 0; i < Math.Min(candidates.Count, targetTokens.Count); i++)
        {
            if (string.Equals(candidates[i], targetTokens[i], StringComparison.Ordinal))
            {
                verified.Add(candidates[i]);
            }
            else
            {
                // First mismatch - accept target token instead
                verified.Add(targetTokens[i]);
                rejected = true;
                break;
            }
        }

        return (verified, rejected);
    }

    private async Task<string?> GenerateTargetTokenAsync(
        string context,
        GenerationOptions? options,
        CancellationToken cancellationToken)
    {
        var singleTokenOptions = CloneWithMaxTokens(options ?? new GenerationOptions(), 1);
        await foreach (var token in _targetModel.GenerateAsync(
            context,
            singleTokenOptions,
            cancellationToken))
        {
            return token;
        }
        return null;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    private static SpeculativeStats CreateEmptyStats() => new()
    {
        TotalTokens = 0,
        DraftTokens = 0,
        AcceptedTokens = 0,
        TargetTokens = 0,
        ElapsedMilliseconds = 0
    };

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _draftModel.DisposeAsync().AsTask().GetAwaiter().GetResult();
        _targetModel.DisposeAsync().AsTask().GetAwaiter().GetResult();
    }

    /// <inheritdoc />
    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;

        await _draftModel.DisposeAsync();
        await _targetModel.DisposeAsync();
    }
}

/// <summary>
/// Builder for creating speculative decoder instances.
/// </summary>
public sealed class SpeculativeDecoderBuilder
{
    private IGeneratorModel? _draftModel;
    private IGeneratorModel? _targetModel;
    private SpeculativeDecodingOptions _options = new();

    /// <summary>
    /// Creates a new builder.
    /// </summary>
    public static SpeculativeDecoderBuilder Create() => new();

    /// <summary>
    /// Sets the draft model.
    /// </summary>
    public SpeculativeDecoderBuilder WithDraftModel(IGeneratorModel draftModel)
    {
        _draftModel = draftModel ?? throw new ArgumentNullException(nameof(draftModel));
        return this;
    }

    /// <summary>
    /// Sets the target model.
    /// </summary>
    public SpeculativeDecoderBuilder WithTargetModel(IGeneratorModel targetModel)
    {
        _targetModel = targetModel ?? throw new ArgumentNullException(nameof(targetModel));
        return this;
    }

    /// <summary>
    /// Sets the speculation length.
    /// </summary>
    public SpeculativeDecoderBuilder WithSpeculationLength(int length)
    {
        _options.SpeculationLength = length;
        return this;
    }

    /// <summary>
    /// Enables adaptive speculation.
    /// </summary>
    public SpeculativeDecoderBuilder WithAdaptiveSpeculation(bool enabled = true)
    {
        _options.AdaptiveSpeculation = enabled;
        return this;
    }

    /// <summary>
    /// Sets the minimum acceptance rate for adaptive speculation.
    /// </summary>
    public SpeculativeDecoderBuilder WithMinAcceptanceRate(double rate)
    {
        _options.MinAcceptanceRate = rate;
        return this;
    }

    /// <summary>
    /// Sets the draft model temperature.
    /// </summary>
    public SpeculativeDecoderBuilder WithDraftTemperature(float temperature)
    {
        _options.DraftTemperature = temperature;
        return this;
    }

    /// <summary>
    /// Builds the speculative decoder.
    /// </summary>
    public SpeculativeDecoder Build()
    {
        if (_draftModel == null)
            throw new InvalidOperationException("Draft model is required. Use WithDraftModel().");
        if (_targetModel == null)
            throw new InvalidOperationException("Target model is required. Use WithTargetModel().");

        return new SpeculativeDecoder(_draftModel, _targetModel, _options);
    }
}
