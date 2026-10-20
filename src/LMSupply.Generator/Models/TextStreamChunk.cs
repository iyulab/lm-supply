namespace LMSupply.Generator.Models;

/// <summary>
/// A chunk from a streaming raw-prompt completion (<see cref="Abstractions.ITextGenerator.GenerateStreamAsync"/>): a text
/// delta, the prompt-reading progress that comes before the first token, or the final chunk that says why the
/// completion ended.
/// </summary>
public sealed record TextStreamChunk
{
    /// <summary>Text delta, or null on a chunk that carries only progress, log probabilities or the ending.</summary>
    public string? Text { get; init; }

    /// <summary>
    /// Why the completion ended, present only on the final chunk: <c>stop</c> (the model or a stop sequence ended it) or
    /// <c>length</c> (the output token limit did).
    /// </summary>
    public string? FinishReason { get; init; }

    /// <summary>
    /// The backend's token counts for the whole completion, present only on the final chunk. Null when the backend does
    /// not report them (the ONNX path does not; llama-server does), or when the stream was cut client-side by the
    /// output-token safety limit.
    /// </summary>
    public ChatTokenUsage? Usage { get; init; }

    /// <summary>
    /// The backend's timings for the whole completion, present only on the final chunk (with <see cref="Usage"/>).
    /// Null under the same conditions as <see cref="Usage"/>, and on llama-server builds that do not send timings.
    /// </summary>
    public GenerationTimings? Timings { get; init; }

    /// <summary>
    /// The log probabilities of the tokens the backend sent with this chunk, when <see cref="GenerationOptions.TopLogprobs"/>
    /// asked for them; null otherwise. They follow the generated tokens, not the text: a chunk whose text a reasoning
    /// filter held back can still carry them.
    /// </summary>
    public IReadOnlyList<TokenLogprob>? Logprobs { get; init; }

    /// <summary>
    /// How far the backend has read the prompt, on the chunks a stream yields before the first token: when it starts
    /// reading and after each batch. Such a chunk carries nothing else. Null on every other chunk, on the ONNX path (it
    /// reads the prompt in one step and reports nothing meanwhile), and on llama-server builds that do not report it.
    /// </summary>
    public PromptProgress? PromptProgress { get; init; }
}
