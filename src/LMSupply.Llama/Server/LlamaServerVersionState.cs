using System.Text.Json.Serialization;

namespace LMSupply.Llama.Server;

/// <summary>
/// Persisted state for a llama-server installation.
/// </summary>
/// <remarks>
/// A build is identified by its version and the backend directory it was extracted under; its location is
/// <c>&lt;cache root&gt;/&lt;version&gt;/&lt;build backend&gt;</c>, resolved against the cache root in use when the state
/// is read (<see cref="LlamaServerStateManager.GetBuildDirectory"/>). The state therefore survives a moved cache root.
/// </remarks>
public sealed class LlamaServerVersionState
{
    /// <summary>
    /// Currently installed version (e.g., "b7898").
    /// </summary>
    [JsonPropertyName("installedVersion")]
    public required string InstalledVersion { get; set; }

    /// <summary>
    /// Backend directory the installed build lives under (e.g. "cuda12", or "cpu" when a GPU request fell back to the
    /// CPU build). Null means <see cref="Backend"/>.
    /// </summary>
    [JsonPropertyName("installedBuildBackend")]
    public string? InstalledBuildBackend { get; set; }

    /// <summary>
    /// Latest known version from GitHub (e.g., "b7900").
    /// </summary>
    [JsonPropertyName("latestKnownVersion")]
    public string? LatestKnownVersion { get; set; }

    /// <summary>
    /// When the latest version was last checked.
    /// </summary>
    [JsonPropertyName("lastVersionCheck")]
    public DateTimeOffset LastVersionCheck { get; set; }

    /// <summary>
    /// Version that has been downloaded but not yet activated.
    /// </summary>
    [JsonPropertyName("pendingVersion")]
    public string? PendingVersion { get; set; }

    /// <summary>
    /// Backend directory the pending build lives under. Null means <see cref="Backend"/>.
    /// </summary>
    [JsonPropertyName("pendingBuildBackend")]
    public string? PendingBuildBackend { get; set; }

    /// <summary>
    /// Whether an update is ready to be applied.
    /// </summary>
    [JsonPropertyName("updateReady")]
    public bool UpdateReady { get; set; }

    /// <summary>
    /// Previous versions kept for rollback (newest first).
    /// </summary>
    [JsonPropertyName("previousVersions")]
    public List<VersionEntry> PreviousVersions { get; set; } = [];

    /// <summary>
    /// Versions that failed to load (will be skipped in future checks).
    /// </summary>
    [JsonPropertyName("failedVersions")]
    public HashSet<string> FailedVersions { get; set; } = [];

    /// <summary>
    /// The GPU backend this state applies to.
    /// </summary>
    [JsonPropertyName("backend")]
    public string Backend { get; set; } = "vulkan";

    // Absolute directories, written for LMSupply releases before 0.122 that share the cache (their schema requires
    // these fields and resolves builds from them; without them they discard the whole state file and re-adopt one
    // backend). Never read here except to migrate a file those releases wrote: the build backend is the last segment.
    // TODO: remove once every iyulab consumer that shares a cache with another pins LMSupply 0.122 or later.
    [JsonInclude]
    [JsonPropertyName("installedPath")]
    internal string? LegacyInstalledPath { get; set; }

    [JsonInclude]
    [JsonPropertyName("pendingPath")]
    internal string? LegacyPendingPath { get; set; }
}

/// <summary>
/// Entry for a previous version kept for rollback.
/// </summary>
public sealed class VersionEntry
{
    /// <summary>
    /// Version string (e.g., "b7898").
    /// </summary>
    [JsonPropertyName("version")]
    public required string Version { get; set; }

    /// <summary>
    /// Backend directory the build lives under. Null means the owning state's <see cref="LlamaServerVersionState.Backend"/>.
    /// </summary>
    [JsonPropertyName("buildBackend")]
    public string? BuildBackend { get; set; }

    /// <summary>
    /// When this version was installed.
    /// </summary>
    [JsonPropertyName("installedAt")]
    public DateTimeOffset InstalledAt { get; set; }

    // See LlamaServerVersionState.LegacyInstalledPath.
    [JsonInclude]
    [JsonPropertyName("path")]
    internal string? LegacyPath { get; set; }
}

/// <summary>
/// Root object for the persisted state file.
/// </summary>
public sealed class LlamaServerStateFile
{
    /// <summary>
    /// Schema version: 2 records builds by version and backend directory (1 recorded absolute paths).
    /// </summary>
    [JsonPropertyName("schemaVersion")]
    public int SchemaVersion { get; set; } = 2;

    /// <summary>
    /// State entries keyed by "{backend}|{platform}" (e.g., "vulkan|win-x64").
    /// </summary>
    [JsonPropertyName("entries")]
    public Dictionary<string, LlamaServerVersionState> Entries { get; set; } = [];
}
