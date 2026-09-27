namespace LMSupply.Download;

/// <summary>One repository file a download fetches, at the length the repository lists for it.</summary>
/// <param name="Path">The file's path in the repository (e.g. <c>onnx/encoder_model_int8.onnx</c>).</param>
/// <param name="SizeBytes">The length the repository listing gives for the file.</param>
public sealed record PlannedFile(string Path, long SizeBytes);

/// <summary>
/// The files a download would fetch into an empty cache, chosen by the same rules the download itself uses —
/// what a consent screen or a disk/bandwidth budget needs before anything is fetched.
/// </summary>
public sealed class DownloadPlan
{
    /// <summary>The HuggingFace repository ID.</summary>
    public required string RepoId { get; init; }

    /// <summary>The revision the plan was made for.</summary>
    public required string Revision { get; init; }

    /// <summary>The files, in download order.</summary>
    public required IReadOnlyList<PlannedFile> Files { get; init; }

    /// <summary>Bytes of all <see cref="Files"/> — the whole download, whatever the cache already holds.</summary>
    public long TotalBytes => Files.Sum(f => f.SizeBytes);
}
