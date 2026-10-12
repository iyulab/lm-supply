using System.Globalization;

namespace LMSupply;

/// <summary>
/// One report of what a download or a load is doing: which file (<see cref="FileName"/>), which step
/// (<see cref="Phase"/>), and — when the step counts something — how far along it is.
/// </summary>
/// <remarks>
/// The byte fields carry bytes or nothing. A step with nothing to count (<see cref="DownloadPhase.Preparing"/>,
/// <see cref="DownloadPhase.Loading"/>, a download whose size the server did not announce) leaves
/// <see cref="TotalBytes"/> at zero and is <see cref="IsIndeterminate"/>.
/// </remarks>
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
    /// Gets the progress of the current file as a percentage (0-100): 100 once the file is
    /// <see cref="DownloadPhase.Complete"/>, otherwise the share of <see cref="TotalBytes"/> done. Zero while
    /// <see cref="IsIndeterminate"/> — that zero is «not measured», not «nothing done».
    /// </summary>
    public double PercentComplete => Phase == DownloadPhase.Complete ? 100
        : TotalBytes > 0 ? (double)BytesDownloaded / TotalBytes * 100 : 0;

    /// <summary>
    /// Gets whether this report measures nothing: the step is under way, and there is no total to be a share of.
    /// True for <see cref="DownloadPhase.Preparing"/> and <see cref="DownloadPhase.Loading"/>, and for a download
    /// whose size is not known. Show such a step as busy, not as a percentage; <see cref="PercentComplete"/> and
    /// <see cref="OverallPercentComplete"/> are zero for it.
    /// </summary>
    public bool IsIndeterminate => Phase != DownloadPhase.Complete && TotalBytes <= 0 && OverallTotalBytes is not > 0;

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
    /// Gets the current phase. <see cref="DownloadPhase.Downloading"/> means bytes are moving: a file found usable in
    /// the cache is reported once as <see cref="DownloadPhase.Complete"/>, never as a download, so «was anything downloaded»
    /// is answered by whether a <see cref="DownloadPhase.Downloading"/> report arrived. A file is reported
    /// <see cref="DownloadPhase.Complete"/> once. <see cref="DownloadPhase.Loading"/> comes after that, when a load
    /// brings the model into memory, and is not followed by a closing report: the load is over when the call that
    /// was given this reporter returns.
    /// </summary>
    public DownloadPhase Phase { get; init; } = DownloadPhase.Downloading;

    /// <summary>
    /// Gets what is being fetched: the model's files, or a runtime a first load provisions once per host and shares
    /// between models (the native ONNX Runtime, llama-server). A load reports both through one progress callback; this
    /// tells them apart, so a long first-run runtime step can be shown as such.
    /// </summary>
    public DownloadKind Kind { get; init; } = DownloadKind.Model;

    /// <summary>
    /// The one report of a file found usable in the cache: <see cref="DownloadPhase.Complete"/>, with the size of
    /// the file on disk.
    /// </summary>
    internal static DownloadProgress CachedFile(string fileName, string path)
    {
        var size = new FileInfo(path).Length;
        return new DownloadProgress
        {
            FileName = fileName,
            BytesDownloaded = size,
            TotalBytes = size,
            Phase = DownloadPhase.Complete
        };
    }

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
/// What a <see cref="DownloadProgress"/> report is about.
/// </summary>
public enum DownloadKind
{
    /// <summary>The model's own files.</summary>
    Model,

    /// <summary>A runtime the load provisions once per host: the native ONNX Runtime, or llama-server.</summary>
    Runtime,
}

/// <summary>
/// Tags every report passing through it as <see cref="DownloadKind.Runtime"/>, so the progress a runtime download shares
/// with the generic file download (byte counts, speed) reaches the caller marked as runtime progress.
/// </summary>
internal sealed class RuntimeDownloadProgress(IProgress<DownloadProgress> inner) : IProgress<DownloadProgress>
{
    public void Report(DownloadProgress value) => inner.Report(value with { Kind = DownloadKind.Runtime });

    public static IProgress<DownloadProgress>? Wrap(IProgress<DownloadProgress>? progress) =>
        progress is null or RuntimeDownloadProgress ? progress : new RuntimeDownloadProgress(progress);
}

/// <summary>
/// The steps a file goes through while it is fetched, and — in a load — brought into memory.
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

    /// <summary>The file is in place: downloaded, or found usable in the cache. Reported once per file.</summary>
    Complete,

    /// <summary>
    /// The files are in place and the model is being brought into memory: an inference server is starting and
    /// reading the weights, or a runtime session is being created. Nothing is counted, so the report carries no
    /// byte counts. It is reported when that work starts — not when a load shares a server that is already
    /// running — and nothing is reported when it ends.
    /// </summary>
    Loading
}
