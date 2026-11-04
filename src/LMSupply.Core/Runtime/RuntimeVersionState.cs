namespace LMSupply.Runtime;

/// <summary>
/// Represents the version state for a runtime package.
/// Stored persistently in runtime-versions.json.
/// </summary>
public sealed class RuntimeVersionState
{
    /// <summary>
    /// Gets or sets the currently installed version.
    /// </summary>
    public required string InstalledVersion { get; set; }

    /// <summary>
    /// Gets or sets the latest known version from NuGet.
    /// </summary>
    public string? LatestKnownVersion { get; set; }

    /// <summary>
    /// Gets or sets the timestamp of the last version check. A new entry starts at the time the runtime was first
    /// recorded, not at <see cref="DateTimeOffset.MinValue"/>: the first background check comes one check interval
    /// after the install, so a first run transfers what its caller agreed to and nothing more.
    /// </summary>
    public DateTimeOffset LastVersionCheck { get; set; }

    /// <summary>
    /// Gets or sets the pending version being downloaded.
    /// </summary>
    public string? PendingVersion { get; set; }

    /// <summary>
    /// Gets or sets whether an update is ready to be applied.
    /// </summary>
    public bool UpdateReady { get; set; }

    /// <summary>
    /// Gets or sets the path to the downloaded update.
    /// </summary>
    public string? UpdateReadyPath { get; set; }

    /// <summary>
    /// Gets or sets previous versions for rollback support.
    /// </summary>
    public List<string> PreviousVersions { get; set; } = [];

    /// <summary>
    /// Gets or sets versions that failed to load (skipped on next check).
    /// </summary>
    public HashSet<string> FailedVersions { get; set; } = [];

    /// <summary>
    /// Gets whether an update is available based on current state.
    /// </summary>
    public bool UpdateAvailable =>
        LatestKnownVersion is not null &&
        !string.Equals(InstalledVersion, LatestKnownVersion, StringComparison.OrdinalIgnoreCase) &&
        !FailedVersions.Contains(LatestKnownVersion);
}

/// <summary>
/// Root state container for all runtime packages.
/// Serialized to runtime-versions.json.
/// </summary>
public sealed class RuntimeVersionStateFile
{
    /// <summary>
    /// Schema version for forward compatibility.
    /// </summary>
    public int SchemaVersion { get; set; } = 1;

    /// <summary>
    /// Package states keyed by package key (e.g., "llama-server|vulkan|win-x64").
    /// </summary>
    public Dictionary<string, RuntimeVersionState> Packages { get; set; } = new();
}
