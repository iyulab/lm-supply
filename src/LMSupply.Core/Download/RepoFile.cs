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
