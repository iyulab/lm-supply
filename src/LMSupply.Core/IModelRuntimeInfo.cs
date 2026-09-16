namespace LMSupply;

/// <summary>
/// Common interface for runtime information about loaded models.
/// All domain-specific ModelInfo types should implement this interface
/// to provide consistent access to runtime diagnostics.
/// </summary>
public interface IModelRuntimeInfo
{
    /// <summary>
    /// Gets the model identifier (e.g., "BAAI/bge-small-en-v1.5" or alias like "default").
    /// </summary>
    string ModelId { get; }

    /// <summary>
    /// Gets the local filesystem path to the model files.
    /// </summary>
    string ModelPath { get; }

    /// <summary>
    /// Gets the execution provider that was requested during model loading.
    /// </summary>
    ExecutionProvider RequestedProvider { get; }

    /// <summary>
    /// Gets the list of execution providers actually active in the inference session.
    /// Typically includes the primary provider and CPU as fallback.
    /// </summary>
    IReadOnlyList<string> ActiveProviders { get; }

    /// <summary>
    /// Gets whether GPU acceleration is actually being used.
    /// True if any GPU provider (CUDA, CoreML) is active.
    /// </summary>
    bool IsGpuActive { get; }

    /// <summary>
    /// Gets the estimated memory usage of this model in bytes (if available).
    /// May return null if memory estimation is not implemented for this model type.
    /// </summary>
    long? EstimatedMemoryBytes { get; }
}
