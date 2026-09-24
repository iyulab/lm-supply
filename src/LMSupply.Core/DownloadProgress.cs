using System.Globalization;

namespace LMSupply;

/// <summary>
/// Progress information for model downloads.
/// </summary>
public record DownloadProgress
{
    /// <summary>
    /// Gets the name of the file being downloaded.
    /// </summary>
    public required string FileName { get; init; }

    /// <summary>
    /// Gets the number of bytes of the current file (<see cref="FileName"/>) downloaded so far.
    /// </summary>
    public long BytesDownloaded { get; init; }

    /// <summary>
    /// Gets the size of the current file (<see cref="FileName"/>) in bytes.
    /// </summary>
    public long TotalBytes { get; init; }

    /// <summary>
    /// Gets the bytes done across the whole multi-file download: the files before this one (downloaded or already
    /// cached) plus <see cref="BytesDownloaded"/>. <c>null</c> when the size of some file is not known up front, and
    /// for a single-file download (use <see cref="BytesDownloaded"/>).
    /// </summary>
    public long? OverallBytesDownloaded { get; init; }

    /// <summary>
    /// Gets the size of the whole multi-file download in bytes, when every file's size is known up front;
    /// otherwise <c>null</c>.
    /// </summary>
    public long? OverallTotalBytes { get; init; }

    /// <summary>
    /// Gets the download progress of the current file as a percentage (0-100).
    /// </summary>
    public double PercentComplete => TotalBytes > 0 ? (double)BytesDownloaded / TotalBytes * 100 : 0;

    /// <summary>
    /// Gets the 1-based index of the current file in a multi-file download.
    /// Zero when downloading a single file outside of a multi-file context.
    /// </summary>
    public int CurrentFileIndex { get; init; }

    /// <summary>
    /// Gets the total number of files in a multi-file download.
    /// Zero when downloading a single file outside of a multi-file context.
    /// </summary>
    public int TotalFileCount { get; init; }

    /// <summary>
    /// Gets the overall download progress across all files as a percentage (0-100). Byte-weighted
    /// (<see cref="OverallBytesDownloaded"/> / <see cref="OverallTotalBytes"/>) when every file's size is known; a
    /// file-count approximation only when it is not. <see cref="PercentComplete"/> for a single-file download.
    /// </summary>
    public double OverallPercentComplete => OverallTotalBytes is > 0 && OverallBytesDownloaded is { } done
        ? Math.Min(100.0, (double)done / OverallTotalBytes.Value * 100)
        : TotalFileCount > 0
            ? ((CurrentFileIndex - 1) * 100.0 + PercentComplete) / TotalFileCount
            : PercentComplete;

    /// <summary>
    /// Gets the current download speed in bytes per second.
    /// </summary>
    public double? BytesPerSecond { get; init; }

    /// <summary>
    /// Gets the estimated time remaining for the download.
    /// </summary>
    public TimeSpan? EstimatedRemaining { get; init; }

    /// <summary>
    /// Gets the current download phase.
    /// </summary>
    public DownloadPhase Phase { get; init; } = DownloadPhase.Downloading;

    /// <summary>
    /// Gets a human-readable speed string (e.g., "15.3 MB/s").
    /// </summary>
    public string SpeedDisplay => BytesPerSecond switch
    {
        null => "",
        < 1024 => $"{BytesPerSecond:F0} B/s",
        < 1024 * 1024 => $"{BytesPerSecond / 1024:F1} KB/s",
        _ => $"{BytesPerSecond / (1024 * 1024):F1} MB/s"
    };

    /// <summary>
    /// Gets a human-readable ETA string (e.g., "2m 30s").
    /// </summary>
    public string EtaDisplay => EstimatedRemaining switch
    {
        null => "",
        { TotalSeconds: < 60 } eta => $"{eta.Seconds}s",
        { TotalMinutes: < 60 } eta => $"{eta.Minutes}m {eta.Seconds}s",
        _ => EstimatedRemaining.Value.ToString(@"h\h\ m\m", CultureInfo.InvariantCulture)
    };
}

/// <summary>
/// Phases of the download process.
/// </summary>
public enum DownloadPhase
{
    /// <summary>Preparing to download (resolving URLs, checking cache).</summary>
    Preparing,

    /// <summary>Actively downloading file data.</summary>
    Downloading,

    /// <summary>Extracting downloaded archive.</summary>
    Extracting,

    /// <summary>Verifying downloaded content.</summary>
    Verifying,

    /// <summary>Finalizing download (moving to cache, cleanup).</summary>
    Finalizing,

    /// <summary>Download complete.</summary>
    Complete
}
