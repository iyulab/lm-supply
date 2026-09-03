namespace LMSupply.Generator.Abstractions;

/// <summary>
/// Extends <see cref="IGeneratorModelFactory"/> with the cache-path resolution the ONNX Runtime
/// GenAI backend needs before a model is downloaded (e.g. to decide whether a download is
/// required). Implemented by <c>OnnxGeneratorModelFactory</c> in the LMSupply.Generator.Onnx
/// package.
/// </summary>
public interface IOnnxGeneratorModelFactory : IGeneratorModelFactory, IDisposable
{
    /// <summary>
    /// Resolves the local cache path for a model, following HuggingFace cache directory structure,
    /// without requiring the model to already be downloaded.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    string GetModelCachePath(string modelId);
}
