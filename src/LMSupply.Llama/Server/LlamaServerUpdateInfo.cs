namespace LMSupply.Llama.Server;

/// <summary>
/// Information about the current llama-server installation and update status.
/// </summary>
public sealed record LlamaServerUpdateInfo
{
    /// <summary>
    /// Currently installed version.
    /// </summary>
    public required string InstalledVersion { get; init; }

    /// <summary>
    /// Latest available version (if known).
    /// </summary>
    public string? LatestVersion { get; init; }

    /// <summary>
    /// Whether an update is available.
    /// </summary>
    public bool UpdateAvailable => LatestVersion != null &&
        !string.Equals(InstalledVersion, LatestVersion, StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Whether an update has been downloaded and is ready to apply.
    /// </summary>
    public bool UpdateReady { get; init; }

    /// <summary>
    /// Version of the pending update (if any).
    /// </summary>
    public string? PendingVersion { get; init; }

    /// <summary>
    /// When the latest version was last checked.
    /// </summary>
    public DateTimeOffset LastChecked { get; init; }

    /// <summary>
    /// The GPU backend being used.
    /// </summary>
    public required LlamaServerBackend Backend { get; init; }

    /// <summary>
    /// Path to the installed llama-server executable.
    /// </summary>
    public required string ServerPath { get; init; }

    /// <summary>
    /// Gets a human-readable summary.
    /// </summary>
    public string GetSummary()
    {
        var status = UpdateReady ? "Update ready"
            : UpdateAvailable ? $"Update available ({LatestVersion})"
            : "Up to date";

        return $"llama-server {InstalledVersion} ({Backend}) - {status}";
    }
}

/// <summary>The llama-server a load runs: its executable, its backend and, when known, its build.</summary>
/// <param name="ServerPath">Path to the llama-server executable.</param>
/// <param name="Backend">The backend the build serves.</param>
/// <param name="Version">The build (for example <c>b7898</c>), or null when it could not be read.</param>
public sealed record LlamaServerInstallation(string ServerPath, LlamaServerBackend Backend, string? Version);

/// <summary>
/// Result of an update check or apply operation.
/// </summary>
public sealed record LlamaServerUpdateResult
{
    /// <summary>
    /// Whether an update was applied.
    /// </summary>
    public bool Updated { get; init; }

    /// <summary>
    /// Previous version (before update).
    /// </summary>
    public string? PreviousVersion { get; init; }

    /// <summary>
    /// New version (after update).
    /// </summary>
    public string? NewVersion { get; init; }

    /// <summary>
    /// Path to the llama-server executable.
    /// </summary>
    public required string ServerPath { get; init; }

    /// <summary>
    /// The GPU backend.
    /// </summary>
    public required LlamaServerBackend Backend { get; init; }

    /// <summary>
    /// Error message if the update failed.
    /// </summary>
    public string? Error { get; init; }

    /// <summary>
    /// Whether the operation succeeded.
    /// </summary>
    public bool Success => Error == null;

    /// <summary>
    /// Creates a successful result with no update.
    /// </summary>
    public static LlamaServerUpdateResult NoUpdate(string serverPath, LlamaServerBackend backend, string version)
        => new()
        {
            Updated = false,
            ServerPath = serverPath,
            Backend = backend,
            PreviousVersion = version,
            NewVersion = version
        };

    /// <summary>
    /// Creates a successful result with an update.
    /// </summary>
    public static LlamaServerUpdateResult WithUpdate(
        string serverPath,
        LlamaServerBackend backend,
        string previousVersion,
        string newVersion)
        => new()
        {
            Updated = true,
            ServerPath = serverPath,
            Backend = backend,
            PreviousVersion = previousVersion,
            NewVersion = newVersion
        };

    /// <summary>
    /// Creates a failed result.
    /// </summary>
    // A local-only acquisition found no server in the cache (LlamaServerUpdateService.AcquireServerAsync throws ModelNotFoundException for it).
    internal bool NotCached { get; init; }

    internal static LlamaServerUpdateResult NotInCache(LlamaServerBackend backend, string error)
        => Failed(string.Empty, backend, error) with { NotCached = true };

    public static LlamaServerUpdateResult Failed(string serverPath, LlamaServerBackend backend, string error)
        => new()
        {
            Updated = false,
            ServerPath = serverPath,
            Backend = backend,
            Error = error
        };
}
