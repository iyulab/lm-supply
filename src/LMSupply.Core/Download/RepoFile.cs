using System.Text.Json.Serialization;

namespace LMSupply.Core.Download;

/// <summary>
/// Represents a file or directory entry in a HuggingFace repository.
/// </summary>
public sealed class RepoFile
{
    /// <summary>
    /// The relative path of the file within the repository.
    /// </summary>
    [JsonPropertyName("path")]
    public required string Path { get; init; }

    /// <summary>
    /// The type of entry: "file" or "directory".
    /// </summary>
    [JsonPropertyName("type")]
    public required string Type { get; init; }

    /// <summary>
    /// The size of the file in bytes (only for files).
    /// </summary>
    [JsonPropertyName("size")]
    public long Size { get; init; }

    /// <summary>
    /// The OID (Object ID) of the file, typically a hash.
    /// </summary>
    [JsonPropertyName("oid")]
    public string? Oid { get; init; }

    /// <summary>
    /// Git LFS details of the file, or <see langword="null"/> when the file is stored in Git directly.
    /// </summary>
    [JsonPropertyName("lfs")]
    public RepoFileLfs? Lfs { get; init; }

    /// <summary>
    /// The name of the file's blob in the Hugging Face hub cache: the SHA-256 of the content for a Git LFS
    /// file, otherwise the Git blob id — the entity tag the hub answers for the file.
    /// </summary>
    [JsonIgnore]
    public string? BlobId => Lfs?.Oid ?? Oid;

    /// <summary>
    /// Whether this entry is a file.
    /// </summary>
    [JsonIgnore]
    public bool IsFile => Type.Equals("file", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Whether this entry is a directory.
    /// </summary>
    [JsonIgnore]
    public bool IsDirectory => Type.Equals("directory", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Gets the file name without the directory path.
    /// </summary>
    [JsonIgnore]
    public string FileName => System.IO.Path.GetFileName(Path);

    /// <summary>
    /// Gets the directory containing this file, or null if in root.
    /// </summary>
    [JsonIgnore]
    public string? Directory
    {
        get
        {
            var lastSlash = Path.LastIndexOf('/');
            return lastSlash > 0 ? Path[..lastSlash] : null;
        }
    }
}

/// <summary>
/// The Git LFS entry of a repository file: the SHA-256 of its content and its length.
/// </summary>
public sealed class RepoFileLfs
{
    /// <summary>
    /// The SHA-256 of the file's content, in lower-case hexadecimal.
    /// </summary>
    [JsonPropertyName("oid")]
    public string? Oid { get; init; }

    /// <summary>
    /// The length of the file's content in bytes.
    /// </summary>
    [JsonPropertyName("size")]
    public long Size { get; init; }
}
