namespace LMSupply.Decider;

/// <summary>
/// The kind of a <see cref="DecisionQuestion"/>: what shape its answer takes.
/// </summary>
public enum DecisionKind
{
    /// <summary>One of the options the caller gives (or none of them, when allowed).</summary>
    Choice,

    /// <summary>Yes or no.</summary>
    YesNo,

    /// <summary>A level on an ordered scale of 2 to 10 levels, lowest first.</summary>
    Score,
}

/// <summary>
/// A question about a state, with the possible answers the caller allows. Create one with <see cref="Choice"/>,
/// <see cref="YesNo"/> or <see cref="Score"/>; questions are answered independently of each other.
/// </summary>
/// <remarks>
/// A decision is one step of reading, not of reasoning: a question whose answer needs arithmetic or several steps of
/// inference (comparing figures, combining conditions) is answered better by generating text.
/// </remarks>
public abstract class DecisionQuestion
{
    /// <summary>The fewest levels a <see cref="ScoreQuestion"/> takes.</summary>
    public const int MinScoreLevels = 2;

    /// <summary>The most levels a <see cref="ScoreQuestion"/> takes.</summary>
    public const int MaxScoreLevels = 10;

    private protected DecisionQuestion(string instructions)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(instructions);
        Instructions = instructions;
    }

    /// <summary>The question, in the language of the state.</summary>
    public string Instructions { get; }

    /// <summary>What shape the answer takes.</summary>
    public abstract DecisionKind Kind { get; }

    /// <summary>
    /// A question answered by one of <paramref name="options"/>.
    /// </summary>
    /// <param name="instructions">The question.</param>
    /// <param name="options">The possible answers, distinct and non-empty; at least two, or one with <paramref name="allowNone"/>.</param>
    /// <param name="descriptions">Optional: what an option means, by option. Every key must be one of <paramref name="options"/>.</param>
    /// <param name="allowNone">
    /// True: «none of these» is a possible answer too. It is reported as <see cref="Decision.Choice"/> <see langword="null"/>
    /// with its probability in <see cref="Decision.NoneProbability"/>.
    /// </param>
    public static ChoiceQuestion Choice(
        string instructions,
        IReadOnlyList<string> options,
        IReadOnlyDictionary<string, string>? descriptions = null,
        bool allowNone = false)
        => new(instructions, options, descriptions, allowNone);

    /// <summary>
    /// A question answered yes or no.
    /// </summary>
    /// <param name="instructions">The question.</param>
    /// <param name="trueMeans">Optional: what a yes means.</param>
    /// <param name="falseMeans">Optional: what a no means.</param>
    public static YesNoQuestion YesNo(string instructions, string? trueMeans = null, string? falseMeans = null)
        => new(instructions, trueMeans, falseMeans);

    /// <summary>
    /// A question answered by a level of an ordered scale.
    /// </summary>
    /// <param name="instructions">The question.</param>
    /// <param name="levels">What each level means, lowest first: 2 to 10 distinct, non-empty descriptions.</param>
    public static ScoreQuestion Score(string instructions, IReadOnlyList<string> levels)
        => new(instructions, levels);
}

/// <summary>A question answered by one of the caller's options. See <see cref="DecisionQuestion.Choice"/>.</summary>
public sealed class ChoiceQuestion : DecisionQuestion
{
    internal ChoiceQuestion(
        string instructions,
        IReadOnlyList<string> options,
        IReadOnlyDictionary<string, string>? descriptions,
        bool allowNone)
        : base(instructions)
    {
        ArgumentNullException.ThrowIfNull(options);
        var list = options.ToArray();
        if (list.Any(string.IsNullOrWhiteSpace))
            throw new ArgumentException("Options must not be empty.", nameof(options));
        if (list.Distinct(StringComparer.Ordinal).Count() != list.Length)
            throw new ArgumentException("Options must be distinct.", nameof(options));
        if (list.Length < (allowNone ? 1 : 2))
            throw new ArgumentException(
                allowNone
                    ? "A choice needs at least one option."
                    : "A choice needs at least two options, or one with allowNone.",
                nameof(options));

        if (descriptions is not null)
        {
            var unknown = descriptions.Keys.FirstOrDefault(k => !list.Contains(k, StringComparer.Ordinal));
            if (unknown is not null)
                throw new ArgumentException($"The description '{unknown}' names no option.", nameof(descriptions));
        }

        Options = list;
        Descriptions = descriptions?.ToDictionary(kv => kv.Key, kv => kv.Value, StringComparer.Ordinal);
        AllowNone = allowNone;
    }

    /// <inheritdoc />
    public override DecisionKind Kind => DecisionKind.Choice;

    /// <summary>The possible answers, in the caller's order.</summary>
    public IReadOnlyList<string> Options { get; }

    /// <summary>What an option means, by option; null when none were given.</summary>
    public IReadOnlyDictionary<string, string>? Descriptions { get; }

    /// <summary>Whether «none of these» is a possible answer.</summary>
    public bool AllowNone { get; }

    /// <summary>The description of <paramref name="option"/>, or null.</summary>
    internal string? DescriptionOf(string option)
        => Descriptions is not null && Descriptions.TryGetValue(option, out var d) && !string.IsNullOrWhiteSpace(d) ? d : null;
}

/// <summary>A question answered yes or no. See <see cref="DecisionQuestion.YesNo"/>.</summary>
public sealed class YesNoQuestion : DecisionQuestion
{
    internal YesNoQuestion(string instructions, string? trueMeans, string? falseMeans)
        : base(instructions)
    {
        TrueMeans = string.IsNullOrWhiteSpace(trueMeans) ? null : trueMeans;
        FalseMeans = string.IsNullOrWhiteSpace(falseMeans) ? null : falseMeans;
    }

    /// <inheritdoc />
    public override DecisionKind Kind => DecisionKind.YesNo;

    /// <summary>What a yes means; null when not given.</summary>
    public string? TrueMeans { get; }

    /// <summary>What a no means; null when not given.</summary>
    public string? FalseMeans { get; }
}

/// <summary>A question answered by a level of an ordered scale. See <see cref="DecisionQuestion.Score"/>.</summary>
public sealed class ScoreQuestion : DecisionQuestion
{
    internal ScoreQuestion(string instructions, IReadOnlyList<string> levels)
        : base(instructions)
    {
        ArgumentNullException.ThrowIfNull(levels);
        var list = levels.ToArray();
        if (list.Length is < MinScoreLevels or > MaxScoreLevels)
            throw new ArgumentException(
                $"A score takes {MinScoreLevels} to {MaxScoreLevels} levels; {list.Length} were given.", nameof(levels));
        if (list.Any(string.IsNullOrWhiteSpace))
            throw new ArgumentException("Levels must not be empty.", nameof(levels));
        if (list.Distinct(StringComparer.Ordinal).Count() != list.Length)
            throw new ArgumentException("Levels must be distinct.", nameof(levels));

        Levels = list;
    }

    /// <inheritdoc />
    public override DecisionKind Kind => DecisionKind.Score;

    /// <summary>What each level means, lowest first.</summary>
    public IReadOnlyList<string> Levels { get; }
}
