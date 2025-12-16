using System.Text.Json;
using System.Text.Json.Serialization;

namespace LMSupply.Runtime;

/// <summary>
/// Represents a manifest of available runtime binaries.
/// The manifest is a static JSON file that bypasses GitHub API rate limits (60 req/hour unauthenticated).
/// </summary>
public sealed class RuntimeManifest
{
    /// <summary>
    /// Gets or sets the manifest version.
    /// </summary>
    [JsonPropertyName("version")]
    public required string Version { get; set; }

    /// <summary>
    /// Gets or sets the last updated timestamp.
    /// </summary>
    [JsonPropertyName("updated")]
    public required DateTimeOffset Updated { get; set; }

    /// <summary>
    /// Gets or sets the available runtime packages.
    /// </summary>
    [JsonPropertyName("packages")]
    public required Dictionary<string, RuntimePackage> Packages { get; set; }

    /// <summary>
    /// Gets a runtime package by name.
    /// </summary>
    public RuntimePackage? GetPackage(string packageName)
    {
        return Packages.TryGetValue(packageName, out var package) ? package : null;
    }

    /// <summary>
    /// Gets all available binary entries for a specific package and RID.
    /// </summary>
    public IEnumerable<RuntimeBinaryEntry> GetBinaries(string packageName, string runtimeIdentifier)
    {
        var package = GetPackage(packageName);
        if (package is null)
            yield break;

        foreach (var version in package.Versions.Values)
        {
            foreach (var binary in version.Binaries)
            {
                if (binary.RuntimeIdentifier.Equals(runtimeIdentifier, StringComparison.OrdinalIgnoreCase))
                    yield return binary;
            }
        }
    }

    /// <summary>
    /// Gets the latest version of a package for a specific RID.
    /// </summary>
    public RuntimeBinaryEntry? GetLatestBinary(string packageName, string runtimeIdentifier)
    {
        var package = GetPackage(packageName);
        if (package is null)
            return null;

        var latestVersion = package.Versions
            .OrderByDescending(v => v.Key, StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault();

        return latestVersion.Value?.Binaries
            .FirstOrDefault(b => b.RuntimeIdentifier.Equals(runtimeIdentifier, StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>
    /// Parses a manifest from JSON string.
    /// </summary>
    public static RuntimeManifest Parse(string json)
    {
        return JsonSerializer.Deserialize<RuntimeManifest>(json, JsonOptions)
            ?? throw new InvalidOperationException("Failed to parse runtime manifest");
    }

    /// <summary>
    /// Parses a manifest from a stream.
    /// </summary>
    public static async Task<RuntimeManifest> ParseAsync(Stream stream, CancellationToken cancellationToken = default)
    {
        return await JsonSerializer.DeserializeAsync<RuntimeManifest>(stream, JsonOptions, cancellationToken)
            ?? throw new InvalidOperationException("Failed to parse runtime manifest");
    }

    /// <summary>
    /// Serializes the manifest to JSON string.
    /// </summary>
    public string ToJson()
    {
        return JsonSerializer.Serialize(this, JsonOptions);
    }

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        WriteIndented = true
    };
}

/// <summary>
/// Represents a runtime package (e.g., "onnxruntime").
/// </summary>
public sealed class RuntimePackage
{
    /// <summary>
    /// Gets or sets the package description.
    /// </summary>
    [JsonPropertyName("description")]
    public string? Description { get; set; }

    /// <summary>
    /// Gets or sets the package homepage URL.
    /// </summary>
    [JsonPropertyName("homepage")]
    public string? Homepage { get; set; }

    /// <summary>
    /// Gets or sets the available versions.
    /// </summary>
    [JsonPropertyName("versions")]
    public required Dictionary<string, RuntimePackageVersion> Versions { get; set; }
}

/// <summary>
/// Represents a specific version of a runtime package.
/// </summary>
public sealed class RuntimePackageVersion
{
    /// <summary>
    /// Gets or sets the release date.
    /// </summary>
    [JsonPropertyName("released")]
    public DateTimeOffset? Released { get; set; }

    /// <summary>
    /// Gets or sets the release notes URL.
    /// </summary>
    [JsonPropertyName("releaseNotes")]
    public string? ReleaseNotes { get; set; }

    /// <summary>
    /// Gets or sets the available binaries for this version.
    /// </summary>
    [JsonPropertyName("binaries")]
    public required List<RuntimeBinaryEntry> Binaries { get; set; }
}

/// <summary>
/// Represents a single binary file entry in the manifest.
/// </summary>
public sealed class RuntimeBinaryEntry
{
    /// <summary>
    /// Gets or sets the runtime identifier (e.g., "win-x64", "linux-x64").
    /// </summary>
    [JsonPropertyName("rid")]
    public required string RuntimeIdentifier { get; set; }

    /// <summary>
    /// Gets or sets the execution provider type (e.g., "cpu", "cuda12", "directml").
    /// </summary>
    [JsonPropertyName("provider")]
    public required string Provider { get; set; }

    /// <summary>
    /// Gets or sets the download URL.
    /// </summary>
    [JsonPropertyName("url")]
    public required string Url { get; set; }

    /// <summary>
    /// Gets or sets the file name.
    /// </summary>
    [JsonPropertyName("fileName")]
    public required string FileName { get; set; }

    /// <summary>
    /// Gets or sets the file size in bytes.
    /// </summary>
    [JsonPropertyName("size")]
    public required long Size { get; set; }

    /// <summary>
    /// Gets or sets the SHA256 checksum.
    /// </summary>
    [JsonPropertyName("sha256")]
    public required string Sha256 { get; set; }

    /// <summary>
    /// Gets or sets optional additional files required by this binary.
    /// </summary>
    [JsonPropertyName("dependencies")]
    public List<string>? Dependencies { get; set; }

    /// <summary>
    /// Gets the file size in megabytes.
    /// </summary>
    [JsonIgnore]
    public double SizeMB => Size / (1024.0 * 1024.0);
}
