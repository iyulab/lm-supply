using System.Collections.Concurrent;
using System.Diagnostics;
using LMSupply.Exceptions;

namespace LMSupply;

/// <summary>
/// Abstract base class for all domain model registries.
/// Provides unified resolution for system aliases, user aliases, model IDs, and fallbacks.
/// </summary>
/// <remarks>
/// Resolution order: auto -> user alias -> system alias -> full ID -> short name -> local path -> HF fallback.
/// </remarks>
public abstract class ModelRegistryBase<TModelInfo> : IModelRegistry<TModelInfo>
    where TModelInfo : IModelInfoBase
{
    private readonly Dictionary<string, TModelInfo> _systemAliases
        = new(StringComparer.OrdinalIgnoreCase);

    private readonly ConcurrentDictionary<string, string> _userAliases
        = new(StringComparer.OrdinalIgnoreCase);

    private readonly Dictionary<string, TModelInfo> _modelsById
        = new(StringComparer.OrdinalIgnoreCase);

    private readonly Dictionary<string, TModelInfo> _modelsByShortName
        = new(StringComparer.OrdinalIgnoreCase);

    private readonly List<TModelInfo> _allModels = [];

    protected ModelRegistryBase(IEnumerable<TModelInfo> systemModels)
    {
        foreach (var model in systemModels)
        {
            _systemAliases[model.AliasName] = model;

            if (!_modelsById.ContainsKey(model.Id))
                _modelsById[model.Id] = model;

            var shortName = model.Id.Contains('/')
                ? model.Id.Split('/').Last()
                : model.Id;
            if (!_modelsByShortName.ContainsKey(shortName))
                _modelsByShortName[shortName] = model;

            if (!_allModels.Any(m => m.Id.Equals(model.Id, StringComparison.OrdinalIgnoreCase)))
                _allModels.Add(model);
        }
    }

    /// <inheritdoc />
    public TModelInfo Resolve(string modelIdOrAlias)
    {
        if (TryResolve(modelIdOrAlias, out var modelInfo))
            return modelInfo!;

        throw new ModelNotFoundException(
            $"Model '{modelIdOrAlias}' not found in registry.", modelIdOrAlias);
    }

    /// <inheritdoc />
    public bool TryResolve(string modelIdOrAlias, out TModelInfo? modelInfo)
    {
        modelInfo = default;

        if (string.IsNullOrWhiteSpace(modelIdOrAlias))
            return false;

        // Strip variant qualifier if present (e.g., "large:fp16" → "large")
        // Qualifier is handled by the loading layer via LMSupplyOptionsBase.SplitQualifier
        var (baseId, _) = LMSupplyOptionsBase.SplitQualifier(modelIdOrAlias);

        // 1. "auto"
        if (baseId.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            modelInfo = GetAutoModel();
            return true;
        }

        // 2. User alias -> resolve target (no re-entry to user aliases).
        // Invariant: a registered user alias ALWAYS resolves. If the target is not
        // otherwise resolvable (e.g. a "gguf:*" domain identifier), trust the user's
        // explicit mapping and fall back on the TARGET — never on the alias name.
        if (_userAliases.TryGetValue(baseId, out var targetId))
        {
            if (TryResolveInternal(targetId, out modelInfo))
                return true;

            modelInfo = CreateFallbackModelInfo(targetId);
            return true;
        }

        // 3-7. Standard resolution
        return TryResolveInternal(baseId, out modelInfo);
    }

    /// <summary>
    /// Like <see cref="TryResolve"/>, but only what the catalog actually knows: <c>auto</c>, a system or
    /// user alias, a full model id or a short name. A local path or a repository id the catalog has no
    /// entry for is <see langword="false"/> — <see cref="TryResolve"/> answers those with a fallback
    /// entry whose values (dimensions, pooling, length, subfolder) are placeholders, and a loader that
    /// took that placeholder for a declaration read the model wrongly. <paramref name="resolvedId"/> is
    /// the id to load when this returns <see langword="false"/>: the user alias's target if there was
    /// one, otherwise the input without its variant qualifier.
    /// </summary>
    public bool TryResolveCatalog(string modelIdOrAlias, out TModelInfo? modelInfo, out string resolvedId)
    {
        modelInfo = default;
        resolvedId = modelIdOrAlias;

        if (string.IsNullOrWhiteSpace(modelIdOrAlias))
        {
            return false;
        }

        var (baseId, _) = LMSupplyOptionsBase.SplitQualifier(modelIdOrAlias);
        resolvedId = baseId;

        if (baseId.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            modelInfo = GetAutoModel();
            return true;
        }

        if (_userAliases.TryGetValue(baseId, out var targetId))
        {
            resolvedId = targetId;
            return TryResolveCataloged(targetId, out modelInfo);
        }

        return TryResolveCataloged(baseId, out modelInfo);
    }

    /// <summary>Steps 3–5 of resolution: what is in the catalog, and nothing made up.</summary>
    private bool TryResolveCataloged(string modelIdOrAlias, out TModelInfo? modelInfo) =>
        _systemAliases.TryGetValue(modelIdOrAlias, out modelInfo!)
        || _modelsById.TryGetValue(modelIdOrAlias, out modelInfo!)
        || _modelsByShortName.TryGetValue(modelIdOrAlias, out modelInfo!);

    private bool TryResolveInternal(string modelIdOrAlias, out TModelInfo? modelInfo)
    {
        // 3. System alias
        if (_systemAliases.TryGetValue(modelIdOrAlias, out modelInfo!))
            return true;

        // 4. Full model ID
        if (_modelsById.TryGetValue(modelIdOrAlias, out modelInfo!))
            return true;

        // 5. Short name
        if (_modelsByShortName.TryGetValue(modelIdOrAlias, out modelInfo!))
            return true;

        // 6. Local path
        if (IsLocalPath(modelIdOrAlias))
        {
            modelInfo = CreateFallbackModelInfo(modelIdOrAlias);
            return true;
        }

        // 7. HuggingFace repo pattern
        if (modelIdOrAlias.Contains('/'))
        {
            modelInfo = CreateFallbackModelInfo(modelIdOrAlias);
            return true;
        }

        modelInfo = default;
        return false;
    }

    /// <inheritdoc />
    public void RegisterAlias(string aliasName, string targetModelId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(aliasName);
        ArgumentException.ThrowIfNullOrWhiteSpace(targetModelId);

        if (_systemAliases.ContainsKey(aliasName)
            || aliasName.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            throw new AliasConflictException(aliasName);
        }

        if (_userAliases.ContainsKey(targetModelId))
        {
            throw new AliasChainException(aliasName, targetModelId);
        }

        _userAliases[aliasName] = targetModelId;
        Trace.TraceInformation($"[ModelRegistry] Registered user alias '{aliasName}' -> '{targetModelId}'");
    }

    /// <summary>
    /// Gets the target of a user-defined alias, if one is registered under the name.
    /// Local* entry points use this to translate a user alias BEFORE format detection
    /// (gguf prefix, path checks), so detection runs against the target, not the alias.
    /// System aliases are not user aliases and return false.
    /// </summary>
    public bool TryGetUserAliasTarget(string aliasName, out string? targetModelId)
    {
        targetModelId = null;
        if (string.IsNullOrWhiteSpace(aliasName))
            return false;

        return _userAliases.TryGetValue(aliasName, out targetModelId);
    }

    /// <inheritdoc />
    public bool RemoveAlias(string aliasName)
    {
        if (string.IsNullOrWhiteSpace(aliasName))
            return false;

        if (_systemAliases.ContainsKey(aliasName))
            return false;

        return _userAliases.TryRemove(aliasName, out _);
    }

    /// <inheritdoc />
    public IReadOnlyList<AliasInfo> GetAliases()
    {
        var result = new List<AliasInfo>();
        result.Add(new AliasInfo("auto", "(hardware-adaptive)", AliasKind.System));

        foreach (var (name, model) in _systemAliases)
            result.Add(new AliasInfo(name, model.Id, AliasKind.System));

        foreach (var (name, targetId) in _userAliases)
            result.Add(new AliasInfo(name, targetId, AliasKind.User));

        return result;
    }

    /// <inheritdoc />
    public IReadOnlyList<TModelInfo> GetAvailableModels() => _allModels.AsReadOnly();

    /// <summary>
    /// Returns the model info for the "auto" alias, which selects a model based on hardware capabilities.
    /// </summary>
    protected abstract TModelInfo GetAutoModel();

    /// <summary>
    /// Creates a fallback model info for an unknown model ID (e.g., a HuggingFace repo or local path).
    /// </summary>
    protected abstract TModelInfo CreateFallbackModelInfo(string modelId);

    private static bool IsLocalPath(string value)
    {
        return value.Contains('\\')
            || value.Contains(":/")
            || value.Contains(":\\")
            || value.StartsWith('.')
            || value.StartsWith('/');
    }
}
