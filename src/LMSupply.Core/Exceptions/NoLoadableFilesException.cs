namespace LMSupply.Exceptions;

/// <summary>
/// The repository exists, but holds no file this library can load (no GGUF file, no ONNX model). A
/// <see cref="ModelNotFoundException"/>, so a caller that catches that type is unaffected; the subtype lets a probe tell
/// "no such repository" from "a repository with nothing to load".
/// </summary>
internal sealed class NoLoadableFilesException : ModelNotFoundException
{
    public NoLoadableFilesException(string message, string? modelId = null) : base(message, modelId)
    {
    }
}
