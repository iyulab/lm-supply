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
    /// Gets the number of bytes downloaded so far.
    /// </summary>
    public long BytesDownloaded { get; init; }

    /// <summary>
    /// Gets the total number of bytes to download.
    /// </summary>
    public long TotalBytes { get; init; }

    /// <summary>
    /// Gets the download progress as a percentage (0-100).
    /// </summary>
    public double PercentComplete => TotalBytes > 0 ? (double)BytesDownloaded / TotalBytes * 100 : 0;
}
