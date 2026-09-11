namespace LMSupply.Llama.Server;

/// <summary>
/// Options for llama-server auto-update behavior.
/// </summary>
public sealed class LlamaServerUpdateOptions
{
    /// <summary>
    /// Default options instance.
    /// </summary>
    public static LlamaServerUpdateOptions Default { get; } = new();

    /// <summary>
    /// How often to check for new versions.
    /// Default: 24 hours.
    /// </summary>
    public TimeSpan VersionCheckInterval { get; set; } = TimeSpan.FromHours(24);

    /// <summary>
    /// Whether to automatically download updates in the background.
    /// Default: true.
    /// </summary>
    public bool AutoDownloadUpdates { get; set; } = true;

    /// <summary>
    /// Whether to apply updates during WarmupAsync.
    /// If true, WarmupAsync will block until updates are applied.
    /// Default: true.
    /// </summary>
    public bool UpdateOnWarmup { get; set; } = true;

    /// <summary>
    /// Whether to follow llama.cpp's nightly build line instead of its versioned stable line.
    /// llama.cpp publishes versioned releases (<c>vX.Y.Z</c>, each naming the build it was cut from)
    /// as its stable line and marks the per-commit <c>bNNNNN</c> build releases as prereleases.
    /// False (default) resolves "latest" to the build the newest versioned release names; true takes
    /// the newest build release outright. Either way the resolved version is a build tag.
    /// </summary>
    public bool IncludePrerelease { get; set; }

    /// <summary>
    /// Maximum number of previous versions to keep for rollback.
    /// Default: 2.
    /// </summary>
    public int MaxVersionsToKeep { get; set; } = 2;

    /// <summary>
    /// Directory for downloaded llama-server builds and their state file.
    /// Default: null — resolves via <see cref="LMSupplyCachePaths.GetLlamaServerDirectory"/>
    /// (LMSUPPLY_CACHE_DIR env override, then the LMSupply local cache root).
    /// </summary>
    public string? CacheDirectory { get; set; }

    /// <summary>
    /// Timeout for each GitHub API request — release lookups and the nightly-tag pointer. Downloading a
    /// server build is not limited by it. A lookup that runs out of time counts as GitHub being
    /// unreachable: the cached build is used when there is one.
    /// Default: 10 seconds.
    /// </summary>
    public TimeSpan ApiTimeout { get; set; } = TimeSpan.FromSeconds(10);

    /// <summary>
    /// Whether to enable verbose logging.
    /// Default: false.
    /// </summary>
    public bool Verbose { get; set; }

    /// <summary>
    /// Pins an exact llama-server release tag (e.g. "b7898"). When set, version resolution never
    /// calls the GitHub "latest release" API — a cache hit for this version makes zero network
    /// calls, a cache miss downloads exactly this tagged asset.
    /// <see cref="LlamaServerUpdateService.CheckAndApplyUpdateAsync"/> also short-circuits: a
    /// pinned installation never re-checks for or auto-applies a newer version. Ignored if
    /// <see cref="ServerBinaryPath"/> is also set.
    /// Default: null (resolves "latest" as before).
    /// </summary>
    public string? PinnedVersion { get; set; }

    /// <summary>
    /// Points directly at an already-provisioned llama-server executable. When set, acquisition
    /// (both the "latest version" network call and any download) is skipped entirely — the given
    /// path is used as-is. Takes precedence over <see cref="PinnedVersion"/>.
    /// Default: null (binary is acquired/cached as before).
    /// </summary>
    public string? ServerBinaryPath { get; set; }
}
