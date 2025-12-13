namespace LocalAI.Generator.Abstractions;

/// <summary>
/// Represents a loaded text generation model.
/// </summary>
public interface IGeneratorModel : ITextGenerator
{
    /// <summary>
    /// Gets the maximum context length supported by the model.
    /// </summary>
    int MaxContextLength { get; }

    /// <summary>
    /// Gets the chat formatter for this model.
    /// </summary>
    IChatFormatter ChatFormatter { get; }

    /// <summary>
    /// Gets information about the model.
    /// </summary>
    GeneratorModelInfo GetModelInfo();
}

/// <summary>
/// Information about a generator model.
/// </summary>
/// <param name="ModelId">The model identifier.</param>
/// <param name="ModelPath">The local path to the model files.</param>
/// <param name="MaxContextLength">Maximum context length.</param>
/// <param name="ChatFormat">The chat format name.</param>
/// <param name="ExecutionProvider">The execution provider being used.</param>
public readonly record struct GeneratorModelInfo(
    string ModelId,
    string ModelPath,
    int MaxContextLength,
    string ChatFormat,
    string ExecutionProvider);
