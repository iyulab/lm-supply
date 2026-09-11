namespace LMSupply.Pool;

/// <summary>
/// Configuration options for the model pool.
/// </summary>
public sealed class ModelPoolOptions
{
    /// <summary>
    /// Maximum memory to allocate for models (in bytes).
    /// If null, uses available GPU or system memory.
    /// </summary>
    public long? MaxMemoryBytes { get; set; }

    /// <summary>
    /// Safety margin for memory calculations (0.0-1.0).
    /// Defaults to 0.2 (20% buffer).
    /// </summary>
    public double MemorySafetyMargin { get; set; } = 0.2;

    /// <summary>
    /// Maximum number of models to keep loaded simultaneously. Loading one more unloads the least
    /// recently used model first, before the memory check (which may unload more). Must be at least 1.
    /// Defaults to 2.
    /// </summary>
    public int MaxLoadedModels { get; set; } = 2;
}
