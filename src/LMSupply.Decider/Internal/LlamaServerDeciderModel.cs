using LMSupply.Llama.Server;

namespace LMSupply.Decider.Internal;

/// <summary>
/// A decider over a llama-server: native when the server reports its model can produce <c>decisions</c>, emulated
/// over chat completions otherwise.
/// </summary>
internal sealed class LlamaServerDeciderModel : IDeciderModel
{
    private readonly Func<CancellationToken, ValueTask<LlamaServerClient>> _client;
    private readonly IAsyncDisposable? _owner;
    private readonly IReadOnlyDictionary<DecisionKind, double>? _temperatures;
    // Emulated only: how many tokens the model's tokenizer gives each answer label.
    private readonly IReadOnlyDictionary<string, int> _labelTokenCounts;
    private List<LlamaServerTimings?> _lastTimings = [];
    private bool _disposed;

    private LlamaServerDeciderModel(
        string modelId,
        DecisionPath path,
        IReadOnlyDictionary<string, int> labelTokenCounts,
        Func<CancellationToken, ValueTask<LlamaServerClient>> client,
        IAsyncDisposable? owner,
        IReadOnlyDictionary<DecisionKind, double>? temperatures)
    {
        ModelId = modelId;
        Path = path;
        _labelTokenCounts = labelTokenCounts;
        _client = client;
        _owner = owner;
        _temperatures = temperatures;
        Capabilities = path == DecisionPath.Native
            ? new DeciderCapabilities(MaxChoices: null, SupportsNone: true, SupportsYesNo: true, SupportsScore: true)
            : new DeciderCapabilities(
                MaxChoices: AllSingle(LabelSystem.Letters) && AllSingle(LabelSystem.Digits) ? EmulatedPrompt.MaxLetterChoices
                    : AllSingle(LabelSystem.Digits) ? EmulatedPrompt.MaxDigitChoices : 0,
                SupportsNone: true,
                SupportsYesNo: AllSingle(LabelSystem.YesNo),
                SupportsScore: AllSingle(LabelSystem.Digits));
    }

    /// <summary>
    /// A decider over the server <paramref name="client"/> reaches: asks the server what its model produces, and on
    /// the emulated path how the model's tokenizer spells every answer label.
    /// </summary>
    /// <param name="modelId">The model's identifier.</param>
    /// <param name="client">The client of the server, current when called (a restarted server has a new one).</param>
    /// <param name="owner">Disposed with the decider (the server lease); null when the caller owns the server.</param>
    /// <param name="temperatures">The temperature per question kind, or null.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    internal static async Task<LlamaServerDeciderModel> CreateAsync(
        string modelId,
        Func<CancellationToken, ValueTask<LlamaServerClient>> client,
        IAsyncDisposable? owner,
        IReadOnlyDictionary<DecisionKind, double>? temperatures,
        CancellationToken cancellationToken)
    {
        var server = await client(cancellationToken).ConfigureAwait(false);
        var modalities = await server.GetOutputModalitiesAsync(cancellationToken).ConfigureAwait(false);
        var path = modalities?.Contains("decisions", StringComparer.Ordinal) == true ? DecisionPath.Native : DecisionPath.Emulated;

        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        if (path == DecisionPath.Emulated)
        {
            foreach (var label in EmulatedPrompt.DigitLabels.Concat(EmulatedPrompt.LetterLabels).Concat(EmulatedPrompt.YesNoLabels))
                counts[label] = (await server.TokenizeAsync(label, cancellationToken: cancellationToken).ConfigureAwait(false)).Count;
        }

        return new LlamaServerDeciderModel(modelId, path, counts, client, owner, temperatures);
    }

    /// <inheritdoc />
    public string ModelId { get; }

    /// <inheritdoc />
    public DecisionPath Path { get; }

    /// <inheritdoc />
    public DeciderCapabilities Capabilities { get; }

    /// <summary>The client of the server now (a test sends its own requests through it).</summary>
    internal async ValueTask<LlamaServerClient> CurrentClientAsync(CancellationToken cancellationToken) =>
        await _client(cancellationToken).ConfigureAwait(false);

    /// <summary>The server's timings of each request the last call sent, in order (null where the server sent none).</summary>
    internal IReadOnlyList<LlamaServerTimings?> LastTimings => _lastTimings;

    /// <inheritdoc />
    public async Task<Decision> DecideAsync(
        string state,
        string name,
        DecisionQuestion question,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(question);
        var result = await DecideAsync(state, new Dictionary<string, DecisionQuestion> { [name] = question }, cancellationToken)
            .ConfigureAwait(false);
        return result[name];
    }

    /// <inheritdoc />
    public Task<DecisionResult> DecideAsync(
        string state,
        IReadOnlyDictionary<string, DecisionQuestion> questions,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(state);
        ArgumentNullException.ThrowIfNull(questions);
        if (questions.Count == 0)
            throw new ArgumentException("At least one question is required.", nameof(questions));
        foreach (var (name, question) in questions)
        {
            if (string.IsNullOrEmpty(name))
                throw new ArgumentException("Question names must not be empty.", nameof(questions));
            if (question is null)
                throw new ArgumentException($"Question '{name}' is null.", nameof(questions));
        }

        return Path == DecisionPath.Native
            ? DecideNativeAsync(state, questions, cancellationToken)
            : DecideEmulatedAsync(state, questions, cancellationToken);
    }

    private async Task<DecisionResult> DecideNativeAsync(
        string state,
        IReadOnlyDictionary<string, DecisionQuestion> questions,
        CancellationToken cancellationToken)
    {
        var server = await _client(cancellationToken).ConfigureAwait(false);
        var response = await server.DecideAsync(NativeWire.BuildRequest(state, questions), cancellationToken).ConfigureAwait(false);
        _lastTimings = [null];

        var decisions = questions
            .Select(kv => KeyValuePair.Create(
                kv.Key,
                DecisionMath.Build(kv.Value, NativeWire.ReadAnswer(kv.Key, kv.Value, response), DecisionPath.Native, 1, _temperatures)))
            .ToList();
        return new DecisionResult(DecisionPath.Native, decisions);
    }

    private async Task<DecisionResult> DecideEmulatedAsync(
        string state,
        IReadOnlyDictionary<string, DecisionQuestion> questions,
        CancellationToken cancellationToken)
    {
        // Every question is built and checked before the first request: a question this model cannot take fails the
        // call without spending model time on the others.
        var prepared = questions.Select(kv => (Name: kv.Key, Question: kv.Value, Emulated: EmulatedPrompt.Build(state, kv.Value))).ToList();
        foreach (var (name, _, emulated) in prepared)
            ThrowIfLabelsNotSingleTokens(name, emulated);

        // In order and one at a time: each prompt starts with the same state, so the server reuses the prefix the
        // previous question evaluated.
        var server = await _client(cancellationToken).ConfigureAwait(false);
        var timings = new List<LlamaServerTimings?>(prepared.Count);
        var decisions = new List<KeyValuePair<string, Decision>>(prepared.Count);
        foreach (var (name, question, emulated) in prepared)
        {
            var response = await server.GenerateChatWithToolsAsync(
                [ChatCompletionMessage.User(emulated.Prompt)],
                RequestOptions(emulated.TopLogprobs),
                cancellationToken).ConfigureAwait(false);
            timings.Add(response.Timings);

            var first = response.Choices?.FirstOrDefault()?.Logprobs?.Content?.FirstOrDefault();
            var slots = EmulatedPrompt.Read(name, first, emulated.SlotLabels, out var labelMass);
            decisions.Add(KeyValuePair.Create(name, DecisionMath.Build(question, slots, DecisionPath.Emulated, labelMass, _temperatures)));
        }

        _lastTimings = timings;
        return new DecisionResult(DecisionPath.Emulated, decisions);
    }

    /// <summary>
    /// One token, thinking off, and the candidates' log probabilities. The probabilities are the model's raw next-token
    /// distribution, so the sampling settings do not change them; they are set neutral (temperature 0, no top-k, top-p,
    /// min-p or repeat penalty) so the request carries nothing else.
    /// </summary>
    internal static ChatCompletionOptions RequestOptions(int topLogprobs) => new()
    {
        MaxTokens = 1,
        Temperature = 0,
        TopP = 1,
        TopK = 0,
        MinP = 0,
        RepeatPenalty = 1,
        EnableThinking = false,
        TopLogprobs = topLogprobs,
    };

    private void ThrowIfLabelsNotSingleTokens(string name, EmulatedQuestion emulated)
    {
        foreach (var label in emulated.SlotLabels)
        {
            var count = _labelTokenCounts.TryGetValue(label, out var c) ? c : 0;
            if (count != 1)
            {
                throw new NotSupportedException(
                    $"Question '{name}' needs the answer label '{label}', which this model's tokenizer spells as {count} tokens; " +
                    "an answer label must be a single token to be read from one generated token.");
            }
        }
    }

    private bool AllSingle(LabelSystem system) =>
        EmulatedPrompt.LabelsOf(system).All(l => _labelTokenCounts.TryGetValue(l, out var c) && c == 1);

    /// <inheritdoc />
    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        _disposed = true;
        if (_owner is not null)
            await _owner.DisposeAsync().ConfigureAwait(false);
    }
}
