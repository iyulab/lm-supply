namespace LMSupply.Decider;

/// <summary>
/// A loaded model that answers typed questions about a state. Load one with <see cref="LocalDecider.LoadAsync"/>.
/// </summary>
public interface IDeciderModel : IAsyncDisposable
{
    /// <summary>The model's identifier (the file name or the repository name it was loaded from).</summary>
    string ModelId { get; }

    /// <summary>How this model answers: natively, or read from an instruct model's next-token probabilities.</summary>
    DecisionPath Path { get; }

    /// <summary>What this model can answer.</summary>
    DeciderCapabilities Capabilities { get; }

    /// <summary>
    /// Answers each of <paramref name="questions"/> about <paramref name="state"/>. The questions are independent: an
    /// answer does not depend on the other questions.
    /// </summary>
    /// <param name="state">The content to evaluate, in any language; the questions and options are best in the same one.</param>
    /// <param name="questions">The questions by name, at least one. The result keeps their order.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <exception cref="ArgumentException">No questions, or a question this model cannot take (more options than <see cref="DeciderCapabilities.MaxChoices"/>).</exception>
    /// <exception cref="NotSupportedException">The model's tokenizer cannot give the answer labels a question needs as single tokens.</exception>
    /// <exception cref="DecisionException">An answer could not be read from the model's output.</exception>
    Task<DecisionResult> DecideAsync(
        string state,
        IReadOnlyDictionary<string, DecisionQuestion> questions,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Answers one question about <paramref name="state"/>.
    /// </summary>
    /// <param name="state">The content to evaluate.</param>
    /// <param name="name">The question's name.</param>
    /// <param name="question">The question.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task<Decision> DecideAsync(
        string state,
        string name,
        DecisionQuestion question,
        CancellationToken cancellationToken = default);
}
