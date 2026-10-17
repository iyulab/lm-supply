using LMSupply.Llama.Server;

namespace LMSupply.Decider;

/// <summary>
/// Configuration for <see cref="LocalDecider.LoadAsync"/>.
/// </summary>
public sealed class DeciderOptions : LMSupplyOptionsBase
{
    /// <summary>The context size a load uses when <see cref="ContextSize"/> is not set (capped at the model's own).</summary>
    public const int DefaultContextSize = 8192;

    /// <summary>
    /// Gets or sets the context size, in tokens: the longest state plus question a decision can take. Null:
    /// <see cref="DefaultContextSize"/>, or the model's trained context when it is shorter.
    /// </summary>
    public int? ContextSize { get; set; }

    /// <summary>
    /// Gets or sets whether to disable automatic download. When true the model file and llama-server come from the
    /// cache (or a local path, an external or pinned server build) only, a missing one throws
    /// <see cref="ModelNotFoundException"/>, and no update check runs.
    /// <para>Default: false</para>
    /// </summary>
    public bool DisableAutoDownload { get; set; }

    /// <summary>
    /// Gets or sets llama-server acquisition policy (version pinning, a pre-provisioned binary path, auto-update
    /// behavior). If null, the process-wide default (<see cref="LlamaServerUpdateService.Instance"/>) is used.
    /// </summary>
    public LlamaServerUpdateOptions? ServerUpdateOptions { get; set; }

    /// <summary>
    /// Gets or sets a temperature per question kind, applied to the answer probabilities as p ∝ p^(1/T). T above 1
    /// flattens them, below 1 sharpens them. A kind with a temperature reports <see cref="Decision.Calibrated"/>.
    /// Fitting the values to your data is up to you; this only applies them. Null or a missing kind: unchanged.
    /// </summary>
    public IReadOnlyDictionary<DecisionKind, double>? Temperature { get; set; }

    /// <summary>
    /// Creates a copy of these options.
    /// </summary>
    public DeciderOptions Clone() => new()
    {
        CacheDirectory = CacheDirectory,
        Provider = Provider,
        LogLevel = LogLevel,
        ThreadCount = ThreadCount,
        QuantizationHint = QuantizationHint,
        ContextSize = ContextSize,
        DisableAutoDownload = DisableAutoDownload,
        ServerUpdateOptions = ServerUpdateOptions,
        Temperature = Temperature?.ToDictionary(kv => kv.Key, kv => kv.Value),
    };

    /// <summary>Throws when a value cannot be used.</summary>
    internal void Validate()
    {
        if (ContextSize is <= 0)
            throw new ArgumentOutOfRangeException(nameof(ContextSize), ContextSize, "ContextSize must be positive.");
        foreach (var (kind, t) in Temperature ?? new Dictionary<DecisionKind, double>())
        {
            if (!(t > 0) || double.IsInfinity(t))
                throw new ArgumentOutOfRangeException(nameof(Temperature), t, $"The temperature for {kind} must be a positive number.");
        }
    }
}

/// <summary>
/// What a loaded decider can answer.
/// </summary>
/// <param name="MaxChoices">
/// The most options a choice question can have, or null when the model decides and does not say (native models).
/// </param>
/// <param name="SupportsNone">Whether a choice can allow «none of these».</param>
/// <param name="SupportsYesNo">Whether yes/no questions can be answered.</param>
/// <param name="SupportsScore">Whether score questions can be answered.</param>
public sealed record DeciderCapabilities(int? MaxChoices, bool SupportsNone, bool SupportsYesNo, bool SupportsScore);

/// <summary>
/// A decision could not be read from the model's output: on <see cref="DecisionPath.Emulated"/>, none of the most
/// probable next tokens was an answer label (the model wanted to start with something else entirely); on
/// <see cref="DecisionPath.Native"/>, the server's answer lacked an expected field.
/// </summary>
public sealed class DecisionException : LMSupplyException
{
    /// <summary>Creates the exception.</summary>
    public DecisionException(string message, string? questionName = null, string? firstToken = null)
        : base(message)
    {
        QuestionName = questionName;
        FirstToken = firstToken;
    }

    /// <summary>The name of the question that could not be answered, when known.</summary>
    public string? QuestionName { get; }

    /// <summary>The model's most probable next token, when that is what went wrong.</summary>
    public string? FirstToken { get; }
}
