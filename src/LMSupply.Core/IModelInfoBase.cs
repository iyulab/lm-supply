namespace LMSupply;

/// <summary>
/// Base interface for model information across all LMSupply packages.
/// Defines common properties shared by all model types.
/// </summary>
public interface IModelInfoBase
{
    /// <summary>
    /// Gets the identifier the model is loaded by (typically its HuggingFace repository id). A repository can hold
    /// several models, so several catalog entries may share it; <see cref="CatalogKey"/> tells them apart.
    /// </summary>
    string Id { get; }

    /// <summary>
    /// Identifies the model within its catalog. Entries with the same key are one model under several aliases, and a
    /// registry lists that model once; entries with different keys are different models, even when they share an
    /// <see cref="Id"/>. <see cref="ModelRegistryBase{TModelInfo}"/> resolves the key to its entry, so it names a model
    /// where the id alone cannot.
    /// </summary>
    /// <remarks>
    /// The default is <see cref="Id"/>, which is right when the id names one model. A type whose entries can share an
    /// id (several voices, files or folders of one repository) returns the id followed by <c>:</c> and what tells them
    /// apart, for example <c>org/name:folder/in/the/repository</c>.
    /// </remarks>
    string CatalogKey => Id;

    /// <summary>
    /// Gets the user-friendly alias name for this model (e.g., "default", "fast", "quality").
    /// </summary>
    string AliasName { get; }

    /// <summary>
    /// Gets a human-readable description of the model.
    /// </summary>
    string? Description { get; }

    /// <summary>
    /// Maximum context length in tokens, if applicable.
    /// </summary>
    int? ContextLength => null;
}

/// <summary>
/// Represents an alias registration in a model registry.
/// </summary>
/// <param name="Name">The alias name (e.g., "default", "my-model").</param>
/// <param name="TargetModelId">
/// What the alias resolves to: for a system alias the model's <see cref="IModelInfoBase.CatalogKey"/> (its id, unless
/// several models share that id), for a user alias the target it was registered with.
/// </param>
/// <param name="Kind">Whether this is a system or user alias.</param>
public record AliasInfo(string Name, string TargetModelId, AliasKind Kind);

/// <summary>
/// Distinguishes system-defined aliases from user-defined aliases.
/// </summary>
public enum AliasKind
{
    /// <summary>Built-in alias defined by the package (e.g., "default", "fast").</summary>
    System,
    /// <summary>User-registered alias.</summary>
    User
}

/// <summary>
/// Non-generic alias surface of a model registry. Lets alias tooling
/// (e.g. <see cref="AliasConfiguration"/>) operate on any registry regardless of
/// its model-info type.
/// </summary>
public interface IAliasRegistry
{
    /// <summary>
    /// Registers a user-defined alias. Throws AliasConflictException if the name
    /// conflicts with a system alias, or AliasChainException if the target is a user alias.
    /// </summary>
    void RegisterAlias(string aliasName, string targetModelId);

    /// <summary>
    /// Removes a user-defined alias. Returns false if the alias doesn't exist.
    /// System aliases cannot be removed.
    /// </summary>
    bool RemoveAlias(string aliasName);

    /// <summary>
    /// Gets all registered aliases (system + user).
    /// </summary>
    IReadOnlyList<AliasInfo> GetAliases();
}

/// <summary>
/// Interface for model registries that manage model configurations and aliases.
/// </summary>
/// <typeparam name="TModelInfo">The type of model information managed by this registry.</typeparam>
public interface IModelRegistry<TModelInfo> : IAliasRegistry where TModelInfo : IModelInfoBase
{
    /// <summary>
    /// Resolves a model identifier to its full information. Throws ModelNotFoundException if not found.
    /// </summary>
    TModelInfo Resolve(string modelIdOrAlias);

    /// <summary>
    /// Tries to resolve a model identifier to its full information.
    /// </summary>
    bool TryResolve(string modelIdOrAlias, out TModelInfo? modelInfo);

    /// <summary>
    /// Gets every model of the catalog, once each: a model registered under several aliases is listed as the first of
    /// its entries, and models that share an id are all listed (see <see cref="IModelInfoBase.CatalogKey"/>).
    /// </summary>
    IReadOnlyList<TModelInfo> GetAvailableModels();
}
