using System.Text.Json;
using System.Text.Json.Serialization;
using LMSupply.Core.Download;
using LMSupply.Json;

namespace LMSupply.Download;

public sealed class DownloadManifest
{
    private const string FileName = ".lmsupply-manifest.json";

    private static readonly JsonSerializerOptions s_jsonOptions = new()
    {
        TypeInfoResolver = CoreJsonContext.Default,
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    /// <summary>
    /// Manifests at this version or later record the length the repository listed for each file, so a
    /// file whose length differs is known to be wrong. Version 1 manifests recorded whatever was on disk
    /// and certify nothing; the downloader re-verifies them against the listing.
    /// </summary>
    public const int VerifiedVersion = 2;

    public int Version { get; set; } = 1;
    public DateTimeOffset CompletedAt { get; set; }
    public string? RepoId { get; set; }
    public string? Revision { get; set; }
    public List<ManifestFileEntry> Files { get; set; } = [];

    public static async Task WriteAsync(string directoryPath, DownloadManifest manifest)
    {
        manifest.CompletedAt = DateTimeOffset.UtcNow;
        var path = Path.Combine(directoryPath, FileName);
        var json = JsonSerializer.Serialize(manifest, s_jsonOptions);

        // Retry on transient file lock (e.g., rapid host restart releasing handles).
        await FileIoRetry.ExecuteAsync(() => File.WriteAllTextAsync(path, json));
    }

    public static async Task<DownloadManifest?> ReadAsync(string directoryPath)
    {
        var path = Path.Combine(directoryPath, FileName);
        if (!File.Exists(path))
            return null;

        try
        {
            var json = await File.ReadAllTextAsync(path);
            return JsonSerializer.Deserialize<DownloadManifest>(json, s_jsonOptions);
        }
        catch (Exception e) when (e is JsonException or IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    /// <summary>
    /// Reads the manifest synchronously. Returns null if not found or corrupt.
    /// </summary>
    public static DownloadManifest? Read(string directoryPath)
    {
        var path = Path.Combine(directoryPath, FileName);
        if (!File.Exists(path))
            return null;

        try
        {
            var json = File.ReadAllText(path);
            return JsonSerializer.Deserialize<DownloadManifest>(json, s_jsonOptions);
        }
        catch (Exception e) when (e is JsonException or IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    public static DownloadManifest CreateFromDirectory(
        string directoryPath, string? repoId = null, string? revision = null)
    {
        var files = Directory.GetFiles(directoryPath)
            .Where(f => !f.EndsWith(".part", StringComparison.OrdinalIgnoreCase)
                     && !Path.GetFileName(f).Equals(FileName, StringComparison.OrdinalIgnoreCase))
            .Select(f => new ManifestFileEntry
            {
                Path = Path.GetFileName(f),
                Size = new FileInfo(f).Length
            })
            .ToList();

        return new DownloadManifest
        {
            RepoId = repoId,
            Revision = revision,
            Files = files
        };
    }
}

public sealed class ManifestFileEntry
{
    public string Path { get; set; } = "";
    public long Size { get; set; }
}
