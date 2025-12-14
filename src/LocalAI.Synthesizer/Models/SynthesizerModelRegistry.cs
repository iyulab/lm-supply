namespace LocalAI.Synthesizer.Models;

/// <summary>
/// Registry for managing synthesizer model configurations.
/// </summary>
public sealed class SynthesizerModelRegistry
{
    private readonly Dictionary<string, SynthesizerModelInfo> _models = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, SynthesizerModelInfo> _byId = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Gets the default registry instance with pre-configured models.
    /// </summary>
    public static SynthesizerModelRegistry Default { get; } = CreateDefault();

    private SynthesizerModelRegistry() { }

    private static SynthesizerModelRegistry CreateDefault()
    {
        var registry = new SynthesizerModelRegistry();
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
    public void Register(SynthesizerModelInfo info)
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
    public bool TryGet(string aliasOrId, out SynthesizerModelInfo? info)
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
    public IEnumerable<SynthesizerModelInfo> GetAll() => _models.Values;
}
