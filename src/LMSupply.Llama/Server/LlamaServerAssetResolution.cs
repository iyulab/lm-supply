namespace LMSupply.Llama.Server;

/// <summary>
/// The outcome of looking for a llama-server asset: the asset, or enough of the search to say why there
/// wasn't one.
/// </summary>
/// <remarks>
/// This is a diagnostic carrier, not an error-handling return type — acquisition still throws. It exists
/// because the throw sites had nothing true to say: three unrelated failures (no release resolved, tag did
/// not name a build, release has no asset for this platform) all produced one sentence naming a backend, and
/// the backend it named was the last link of the CPU fallback chain rather than the one asked for. A consumer
/// on a GPU-less machine read that as "there is no CPU build" and spent a round trip on a gap that does not
/// exist.
/// </remarks>
public sealed record LlamaServerAssetResolution
{
    /// <summary>The asset to download, or null when resolution failed.</summary>
    public LlamaServerAsset? Asset { get; init; }

    /// <summary>The platform the search was for.</summary>
    public LlamaServerPlatform Platform { get; init; }

    /// <summary>The architecture the search was for.</summary>
    public LlamaServerArchitecture Architecture { get; init; }

    /// <summary>The backend the caller asked for — not whatever the fallback chain ended on.</summary>
    public LlamaServerBackend RequestedBackend { get; init; }

    /// <summary>Every backend actually tried, in order. Empty when the search never got as far as assets.</summary>
    public IReadOnlyList<LlamaServerBackend> BackendsTried { get; init; } = [];

    /// <summary>The release tag whose assets were searched, or null when no release resolved.</summary>
    public string? ReleaseTag { get; init; }

    /// <summary>Asset names present in that release — the evidence for "none of these match".</summary>
    public IReadOnlyList<string> AvailableAssets { get; init; } = [];

    /// <summary>One clause naming which of the failure modes this was. Null on success.</summary>
    public string? Failure { get; init; }

    internal static LlamaServerAssetResolution Found(LlamaServerAsset asset) => new() { Asset = asset };

    internal static LlamaServerAssetResolution Failed(
        LlamaServerPlatform platform,
        LlamaServerArchitecture architecture,
        LlamaServerBackend requestedBackend,
        string? releaseTag,
        string failure,
        IReadOnlyList<LlamaServerBackend>? backendsTried = null,
        IReadOnlyList<string>? availableAssets = null) => new()
        {
            Platform = platform,
            Architecture = architecture,
            RequestedBackend = requestedBackend,
            ReleaseTag = releaseTag,
            Failure = failure,
            BackendsTried = backendsTried ?? [],
            AvailableAssets = availableAssets ?? [],
        };

    /// <summary>
    /// The failure message. Ordered so the first clause is the thing a reader is most likely to act on, and so
    /// the requested backend can never be confused with a fallback step.
    /// </summary>
    public string Describe()
    {
        if (Asset != null)
            return $"Resolved {Asset.Name} ({Asset.Backend}) from release {Asset.Version}.";

        var parts = new List<string>
        {
            $"Could not acquire llama-server: {Failure}.",
            $"Requested backend: {RequestedBackend}; platform {Platform}/{Architecture}.",
        };

        if (BackendsTried.Count > 0)
            parts.Add($"Backends tried, in order: {string.Join(" -> ", BackendsTried)}.");

        parts.Add(ReleaseTag is null
            ? "No release was searched."
            : $"Release searched: {ReleaseTag}.");

        if (ReleaseTag is not null)
        {
            parts.Add(AvailableAssets.Count == 0
                ? "That release publishes no assets."
                : $"Assets in that release: {string.Join(", ", AvailableAssets)}.");
        }

        return string.Join(" ", parts);
    }
}
