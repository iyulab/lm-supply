namespace LMSupply.Llama.Server;

/// <summary>
/// Which of the acquisition failures a <see cref="LlamaServerAssetResolution"/> describes.
/// </summary>
/// <remarks>
/// <para>
/// The prose in <see cref="LlamaServerAssetResolution.Failure"/> says what happened; this says what
/// KIND of thing happened, which is the part a caller can branch on. A consumer deciding whether to
/// offer local inference on a machine needs to tell "this combination has no build at all" from
/// "the release lookup did not answer this time" - the first is a property of the device, the second
/// is a moment.
/// </para>
/// <para>
/// Before this existed the only way to tell them apart was to match on the message text. A consumer
/// did exactly that, and a message rewrite silently disabled their classifier against the case it
/// was written for, with nothing anywhere going red. Values are assigned explicitly so a future
/// insertion cannot renumber the ones already persisted or logged.
/// </para>
/// </remarks>
public enum LlamaServerAcquisitionFailure
{
    /// <summary>
    /// No release could be resolved - the latest-release lookup returned nothing, or a named
    /// release does not exist. Usually transient or a network condition; retrying can succeed.
    /// </summary>
    ReleaseNotResolved = 1,

    /// <summary>
    /// A release resolved but does not name a build: it publishes no nightly-tag asset and no build
    /// release could be found behind it. An upstream publishing state, not a property of this
    /// machine; retrying later can succeed.
    /// </summary>
    ReleaseTagNotABuild = 2,

    /// <summary>
    /// A build release was searched and none of its assets match this platform and architecture, for
    /// any backend in the fallback chain. This is the one that says something about the DEVICE: no
    /// amount of retrying produces a binary for it from this release.
    /// </summary>
    NoAssetForPlatform = 3
}

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
    /// <remarks>
    /// Prose, for a human reading a log. Branch on <see cref="Reason"/> instead — this wording is
    /// not a stable contract, and treating it as one is what broke a consumer's classifier when it
    /// last changed.
    /// </remarks>
    public string? Failure { get; init; }

    /// <summary>
    /// Which kind of failure this was, or null on success. This is the stable part — see
    /// <see cref="LlamaServerAcquisitionFailure"/>.
    /// </summary>
    public LlamaServerAcquisitionFailure? Reason { get; init; }

    internal static LlamaServerAssetResolution Found(LlamaServerAsset asset) => new() { Asset = asset };

    internal static LlamaServerAssetResolution Failed(
        LlamaServerPlatform platform,
        LlamaServerArchitecture architecture,
        LlamaServerBackend requestedBackend,
        string? releaseTag,
        LlamaServerAcquisitionFailure reason,
        string failure,
        IReadOnlyList<LlamaServerBackend>? backendsTried = null,
        IReadOnlyList<string>? availableAssets = null) => new()
        {
            Platform = platform,
            Architecture = architecture,
            RequestedBackend = requestedBackend,
            ReleaseTag = releaseTag,
            Reason = reason,
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
