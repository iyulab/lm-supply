namespace LMSupply;

/// <summary>
/// Base interface for model information across all LMSupply packages.
/// Defines common properties shared by all model types.
/// </summary>
public interface IModelInfoBase
{
    /// <summary>
    /// Gets the unique identifier for this model (typically HuggingFace repo ID).
    /// </summary>
    string Id { get; }

    /// <summary>
    /// Gets the user-friendly alias for this model (e.g., "default", "fast", "quality").
    /// </summary>
    string Alias { get; }

    /// <summary>
    /// Gets a human-readable description of the model.
    /// </summary>
    string? Description { get; }
}

/// <summary>
/// Interface for model registries that manage model configurations.
/// </summary>
/// <typeparam name="TModelInfo">The type of model information managed by this registry.</typeparam>
public interface IModelRegistry<TModelInfo> where TModelInfo : IModelInfoBase
{
    /// <summary>
    /// Tries to resolve a model identifier to its full information.
    /// </summary>
    /// <param name="modelIdOrAlias">Model ID, alias, or local path.</param>
    /// <param name="modelInfo">The resolved model information, or null if not found.</param>
    /// <returns>True if the model was found, false otherwise.</returns>
    bool TryResolve(string modelIdOrAlias, out TModelInfo? modelInfo);

    /// <summary>
    /// Gets all registered model aliases.
    /// </summary>
    /// <returns>Collection of model aliases.</returns>
    IEnumerable<string> GetAliases();

    /// <summary>
    /// Gets all registered model information.
    /// </summary>
    /// <returns>Collection of model information.</returns>
    IEnumerable<TModelInfo> GetAll();
}
