using System.Collections;

namespace LMSupply.Decider;

/// <summary>
/// How a model answers decisions.
/// </summary>
public enum DecisionPath
{
    /// <summary>
    /// The model is a native decision model: llama-server computes the probabilities of the answers
    /// (<c>/v1/systemone</c>), every question of a call in one request.
    /// </summary>
    Native,

    /// <summary>
    /// The model is an instruct model: each question is a one-token chat completion whose answer labels' next-token
    /// probabilities are read and renormalized over the labels.
    /// </summary>
    Emulated,
}

/// <summary>
/// The answer to one <see cref="DecisionQuestion"/>. The typed accessors match the question's <see cref="Kind"/>:
/// <see cref="Choice"/> for a choice, <see cref="IsTrue"/> and <see cref="Probability"/> for yes/no,
/// <see cref="Score"/> and <see cref="Level"/> for a score; the others throw <see cref="InvalidOperationException"/>.
/// </summary>
public sealed class Decision
{
    private readonly string? _choice;
    private readonly double _probabilityTrue;
    private readonly double _score;
    private readonly int _level;

    internal Decision(
        DecisionKind kind,
        DecisionPath path,
        IReadOnlyDictionary<string, double> probabilities,
        double noneProbability,
        double confidence,
        double labelMass,
        bool calibrated,
        string? choice = null,
        double probabilityTrue = 0,
        double score = 0,
        int level = 0)
    {
        Kind = kind;
        Path = path;
        Probabilities = probabilities;
        NoneProbability = noneProbability;
        Confidence = confidence;
        LabelMass = labelMass;
        Calibrated = calibrated;
        _choice = choice;
        _probabilityTrue = probabilityTrue;
        _score = score;
        _level = level;
    }

    /// <summary>The kind of the question this answers.</summary>
    public DecisionKind Kind { get; }

    /// <summary>How the answer was computed.</summary>
    public DecisionPath Path { get; }

    /// <summary>
    /// A choice: the most probable option, or <see langword="null"/> when «none of these» is the most probable answer.
    /// </summary>
    public string? Choice => Kind == DecisionKind.Choice ? _choice : throw WrongKind(nameof(Choice), DecisionKind.Choice);

    /// <summary>A yes/no question: whether yes is the more probable answer.</summary>
    public bool IsTrue => Kind == DecisionKind.YesNo ? _probabilityTrue > 0.5 : throw WrongKind(nameof(IsTrue), DecisionKind.YesNo);

    /// <summary>A yes/no question: the probability of yes.</summary>
    public double Probability => Kind == DecisionKind.YesNo ? _probabilityTrue : throw WrongKind(nameof(Probability), DecisionKind.YesNo);

    /// <summary>
    /// A score: the expected level, Σ i·p(i) with the first level 0. It can fall between two levels.
    /// </summary>
    public double Score => Kind == DecisionKind.Score ? _score : throw WrongKind(nameof(Score), DecisionKind.Score);

    /// <summary>A score: the most probable level, 0 being the first.</summary>
    public int Level => Kind == DecisionKind.Score ? _level : throw WrongKind(nameof(Level), DecisionKind.Score);

    /// <summary>
    /// The probability of each answer: by option for a choice, by level description for a score, <c>"yes"</c> and
    /// <c>"no"</c> for a yes/no question. With <see cref="NoneProbability"/> they sum to 1.
    /// </summary>
    public IReadOnlyDictionary<string, double> Probabilities { get; }

    /// <summary>The probability of «none of these»; 0 for a question that does not allow it.</summary>
    public double NoneProbability { get; }

    /// <summary>The probability of the most probable answer («none of these» included).</summary>
    public double Confidence { get; }

    /// <summary>
    /// How much of the model's next-token probability fell on the answer labels before renormalization
    /// (<see cref="DecisionPath.Emulated"/>). A low value means the model mostly wanted to say something else, and the
    /// renormalized probabilities rest on little. Always 1 on <see cref="DecisionPath.Native"/>.
    /// </summary>
    public double LabelMass { get; }

    /// <summary>
    /// Whether a temperature from <see cref="DeciderOptions.Temperature"/> was applied to these probabilities. It says
    /// what was done, not that the probabilities are calibrated for your data.
    /// </summary>
    public bool Calibrated { get; }

    private InvalidOperationException WrongKind(string member, DecisionKind kind) =>
        new($"{member} answers a {kind} question; this decision answers a {Kind} question.");
}

/// <summary>
/// The answers to a call's questions, by question name, in the order the questions were given.
/// </summary>
[System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "CA1710:Identifiers should have correct suffix",
    Justification = "A call's result, read by question name; not a general-purpose dictionary.")]
public sealed class DecisionResult : IReadOnlyDictionary<string, Decision>
{
    private readonly IReadOnlyList<KeyValuePair<string, Decision>> _ordered;
    private readonly Dictionary<string, Decision> _byName;

    internal DecisionResult(DecisionPath path, IReadOnlyList<KeyValuePair<string, Decision>> ordered)
    {
        Path = path;
        _ordered = ordered;
        _byName = ordered.ToDictionary(kv => kv.Key, kv => kv.Value, StringComparer.Ordinal);
    }

    /// <summary>How the answers were computed.</summary>
    public DecisionPath Path { get; }

    /// <inheritdoc />
    public Decision this[string key] => _byName[key];

    /// <inheritdoc />
    public IEnumerable<string> Keys => _ordered.Select(kv => kv.Key);

    /// <inheritdoc />
    public IEnumerable<Decision> Values => _ordered.Select(kv => kv.Value);

    /// <inheritdoc />
    public int Count => _ordered.Count;

    /// <inheritdoc />
    public bool ContainsKey(string key) => _byName.ContainsKey(key);

    /// <inheritdoc />
    public bool TryGetValue(string key, [System.Diagnostics.CodeAnalysis.MaybeNullWhen(false)] out Decision value)
        => _byName.TryGetValue(key, out value);

    /// <inheritdoc />
    public IEnumerator<KeyValuePair<string, Decision>> GetEnumerator() => _ordered.GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
