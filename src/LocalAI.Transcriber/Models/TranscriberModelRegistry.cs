namespace LocalAI.Transcriber.Models;

/// <summary>
/// Registry for managing transcriber model configurations.
/// </summary>
public sealed class TranscriberModelRegistry
{
    private readonly Dictionary<string, TranscriberModelInfo> _models = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, TranscriberModelInfo> _byId = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Gets the default registry instance with pre-configured models.
    /// </summary>
    public static TranscriberModelRegistry Default { get; } = CreateDefault();

    private TranscriberModelRegistry() { }

    private static TranscriberModelRegistry CreateDefault()
    {
        var registry = new TranscriberModelRegistry();
        foreach (var model in DefaultModels.All)
        {
            registry.Register(model);
        }
        return registry;
    }

    /// <summary>
    /// Registers a model configuration.
    /// </summary>
    /// <param name="info">The model information to register.</param>
    public void Register(TranscriberModelInfo info)
    {
        _models[info.Alias] = info;
        _byId[info.Id] = info;
    }

    /// <summary>
    /// Tries to get model info by alias or ID.
    /// </summary>
    /// <param name="aliasOrId">The alias or HuggingFace model ID.</param>
    /// <param name="info">The model information if found.</param>
    /// <returns>True if found, false otherwise.</returns>
    public bool TryGet(string aliasOrId, out TranscriberModelInfo? info)
    {
        if (_models.TryGetValue(aliasOrId, out info))
            return true;

        if (_byId.TryGetValue(aliasOrId, out info))
            return true;

        info = null;
        return false;
    }

    /// <summary>
    /// Gets all registered aliases.
    /// </summary>
    /// <returns>Collection of model aliases.</returns>
    public IEnumerable<string> GetAliases() => _models.Keys;

    /// <summary>
    /// Gets all registered model information.
    /// </summary>
    /// <returns>Collection of model information.</returns>
    public IEnumerable<TranscriberModelInfo> GetAll() => _models.Values;
}
