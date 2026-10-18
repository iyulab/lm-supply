using System.Text.Json;
using System.Text.Json.Serialization;
using LMSupply.Core.Download;
using LMSupply.Json;

namespace LMSupply.Download;

/// <summary>
/// The record this library keeps of what it downloaded into a snapshot: the files and the lengths the repository
/// listed for them. It also marks the snapshot as this library's own (see <see cref="CacheManager.DeleteModel"/>).
/// </summary>
/// <remarks>
/// Kept outside the snapshot, at <c>models--{org}--{name}/.lmsupply/manifests/{snapshot}.json</c>
/// (<c>{snapshot}__{subfolder}.json</c> for a subfolder download), so a snapshot holds only repository files — what
/// other Hugging Face tools expect to find there. A manifest inside the directory itself
/// (<c>.lmsupply-manifest.json</c>, written by earlier versions and for a directory outside a hub cache) is still read.
/// </remarks>
public sealed class DownloadManifest
{
    private const string FileName = ".lmsupply-manifest.json";

    /// <summary>The directory in a repository's cache directory that holds this library's own files.</summary>
    internal const string PrivateDirectoryName = ".lmsupply";

    private const string ManifestsDirectoryName = "manifests";

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, SemaphoreSlim> s_writeGates =
        new(StringComparer.OrdinalIgnoreCase);

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

    /// <summary>
    /// Requested files the repository does not have, as the repository listing (or a 404) showed when the manifest
    /// was written — the optional files of a default file list that most repositories lack. A load whose other files
    /// are all cached makes no request for these. Null in manifests written before the record existed; such a load
    /// lists the repository once more and records them. Only honored at <see cref="VerifiedVersion"/> or later.
    /// </summary>
    public List<string>? AbsentFiles { get; set; }

    /// <summary>
    /// Writes the manifest inside <paramref name="directoryPath"/> (<c>.lmsupply-manifest.json</c>) — the layout of
    /// a snapshot named after its revision. A download into the hub layout writes it with
    /// <see cref="WriteForSnapshotAsync"/> instead.
    /// </summary>
    public static async Task WriteAsync(string directoryPath, DownloadManifest manifest, CancellationToken cancellationToken = default)
        => await WriteFileAsync(Path.Combine(directoryPath, FileName), manifest);

    /// <summary>
    /// Writes the manifest for <paramref name="snapshotDir"/> (and <paramref name="subfolder"/>) to the repository's
    /// private directory — see <see cref="GetPath"/>.
    /// </summary>
    internal static async Task WriteForSnapshotAsync(string repoDir, string snapshotDir, string? subfolder, DownloadManifest manifest)
    {
        var path = GetPath(repoDir, snapshotDir, subfolder);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        await WriteFileAsync(path, manifest);
    }

    // Several loads of one model can run at once (in one process or several), and each ends by recording what it
    // found. A record that says what the file already says is not written: a load of a cached model leaves the file
    // alone. Otherwise the new text goes to a file of its own and replaces the manifest in one step, so a reader sees
    // the old manifest or the new one, never a half-written one, and two writers never hold the same file open.
    private static async Task WriteFileAsync(string path, DownloadManifest manifest)
    {
        // Writers in this process take turns per file; writers in other processes are met by the retry below.
        var gate = s_writeGates.GetOrAdd(Path.GetFullPath(path), _ => new SemaphoreSlim(1, 1));
        await gate.WaitAsync().ConfigureAwait(false);
        try
        {
            await WriteFileCoreAsync(path, manifest).ConfigureAwait(false);
        }
        finally
        {
            gate.Release();
        }
    }

    private static async Task WriteFileCoreAsync(string path, DownloadManifest manifest)
    {
        if (ReadFile(path) is { } existing)
        {
            manifest.CompletedAt = existing.CompletedAt;
            if (Serialize(manifest) == Serialize(existing))
                return;
        }

        manifest.CompletedAt = DateTimeOffset.UtcNow;
        var json = Serialize(manifest);
        var temp = $"{path}.{Guid.NewGuid():N}.tmp";
        try
        {
            await File.WriteAllTextAsync(temp, json).ConfigureAwait(false);
            // Retry on a transient lock: another writer replacing the file at the same moment, or a handle a host
            // restart has not released yet.
            await FileIoRetry.ExecuteAsync(() => Replace(temp, path)).ConfigureAwait(false);
        }
        finally
        {
            if (File.Exists(temp))
                File.Delete(temp);
        }
    }

    // Windows refuses a replace while another process is replacing the same file with «access denied», not a sharing
    // violation; it is the same transient condition, so it is retried like one.
    private static void Replace(string source, string destination)
    {
        try
        {
            File.Move(source, destination, overwrite: true);
        }
        catch (UnauthorizedAccessException ex) when (File.Exists(source))
        {
            throw new IOException(ex.Message, ex);
        }
    }

    private static string Serialize(DownloadManifest manifest)
        => JsonSerializer.Serialize(manifest, s_jsonOptions.TypeInfoOf(manifest));

    // Readers share the file with writers and with its replacement (FileShare.Delete): a reader open on the manifest
    // must not make a writer's replace fail.
    private static FileStream OpenShared(string path)
        => new(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);

    private static string ReadSharedText(string path)
    {
        using var reader = new StreamReader(OpenShared(path));
        return reader.ReadToEnd();
    }

    private static async Task<string> ReadSharedTextAsync(string path, CancellationToken cancellationToken)
    {
        using var reader = new StreamReader(OpenShared(path));
        return await reader.ReadToEndAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Reads the manifest of a model directory: for a snapshot (or a subfolder of one) in a hub cache, the one in the
    /// repository's private directory, else the one inside the directory. Returns null if not found or corrupt.
    /// </summary>
    public static async Task<DownloadManifest?> ReadAsync(string directoryPath, CancellationToken cancellationToken = default)
    {
        foreach (var path in CandidatePaths(directoryPath))
        {
            if (!File.Exists(path))
                continue;

            try
            {
                var json = await ReadSharedTextAsync(path, cancellationToken).ConfigureAwait(false);
                return JsonSerializer.Deserialize(json, s_jsonOptions.TypeInfo<DownloadManifest>());
            }
            catch (Exception e) when (e is JsonException or IOException or UnauthorizedAccessException)
            {
                return null;
            }
        }

        return null;
    }

    /// <summary>
    /// Reads the manifest synchronously (see <see cref="ReadAsync"/>). Returns null if not found or corrupt.
    /// </summary>
    public static DownloadManifest? Read(string directoryPath)
    {
        foreach (var path in CandidatePaths(directoryPath))
        {
            if (!File.Exists(path))
                continue;

            try
            {
                var json = ReadSharedText(path);
                return JsonSerializer.Deserialize(json, s_jsonOptions.TypeInfo<DownloadManifest>());
            }
            catch (Exception e) when (e is JsonException or IOException or UnauthorizedAccessException)
            {
                return null;
            }
        }

        return null;
    }

    /// <summary>
    /// Where the manifest of <paramref name="snapshotDir"/> (downloaded with <paramref name="subfolder"/>, if any) is
    /// written: <c>{repoDir}/.lmsupply/manifests/{snapshot}.json</c>, or <c>{snapshot}__{subfolder}.json</c> with
    /// every <c>/</c> of the subfolder replaced by <c>_</c>.
    /// </summary>
    internal static string GetPath(string repoDir, string snapshotDir, string? subfolder)
    {
        var snapshot = Path.GetFileName(Path.TrimEndingDirectorySeparator(snapshotDir));
        var name = string.IsNullOrEmpty(subfolder)
            ? snapshot
            : $"{snapshot}__{subfolder.Trim('/', '\\').Replace('/', '_').Replace('\\', '_')}";
        return Path.Combine(repoDir, PrivateDirectoryName, ManifestsDirectoryName, name + ".json");
    }

    /// <summary>
    /// Whether this library recorded a download into <paramref name="snapshotDir"/>: a manifest for it (for any
    /// subfolder) in the repository's private directory, or one inside the snapshot from an earlier version.
    /// </summary>
    internal static bool ExistsForSnapshot(string repoDir, string snapshotDir)
    {
        if (File.Exists(Path.Combine(snapshotDir, FileName)))
            return true;

        var manifests = Path.Combine(repoDir, PrivateDirectoryName, ManifestsDirectoryName);
        if (!Directory.Exists(manifests))
            return false;

        var snapshot = Path.GetFileName(Path.TrimEndingDirectorySeparator(snapshotDir));
        try
        {
            return File.Exists(Path.Combine(manifests, snapshot + ".json"))
                   || Directory.EnumerateFiles(manifests, snapshot + "__*.json").Any();
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>
    /// The manifest files that describe <paramref name="snapshotDir"/> in the repository's private directory.
    /// </summary>
    internal static IEnumerable<string> EnumerateForSnapshot(string repoDir, string snapshotDir)
    {
        var manifests = Path.Combine(repoDir, PrivateDirectoryName, ManifestsDirectoryName);
        if (!Directory.Exists(manifests))
            return [];

        var snapshot = Path.GetFileName(Path.TrimEndingDirectorySeparator(snapshotDir));
        return Directory.EnumerateFiles(manifests, snapshot + ".json")
            .Concat(Directory.EnumerateFiles(manifests, snapshot + "__*.json"))
            .ToList();
    }

    /// <summary>Reads a manifest file. Returns null if not found or corrupt.</summary>
    internal static DownloadManifest? ReadFile(string path)
    {
        try
        {
            return File.Exists(path)
                ? JsonSerializer.Deserialize(ReadSharedText(path), s_jsonOptions.TypeInfo<DownloadManifest>())
                : null;
        }
        catch (Exception e) when (e is JsonException or IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    /// <summary>The name of the manifest file kept inside a directory (earlier versions, and outside a hub cache).</summary>
    internal static string LegacyFileName => FileName;

    // The private manifest first (when the directory is a snapshot, or a subfolder of one, in a hub cache), then the
    // one inside the directory.
    private static IEnumerable<string> CandidatePaths(string directoryPath)
    {
        if (TryLocateInCache(directoryPath, out var repoDir, out var snapshotDir, out var subfolder))
            yield return GetPath(repoDir, snapshotDir, subfolder);

        yield return Path.Combine(directoryPath, FileName);
    }

    /// <summary>
    /// Splits a directory inside a hub cache into its repository directory (<c>models--{org}--{name}</c>), its
    /// snapshot (<c>snapshots/{name}</c>) and the subfolder beneath the snapshot (<see langword="null"/> at its root).
    /// False for a directory that is not inside a snapshot.
    /// </summary>
    internal static bool TryLocateInCache(string directoryPath, out string repoDir, out string snapshotDir, out string? subfolder)
    {
        repoDir = snapshotDir = string.Empty;
        subfolder = null;

        string full;
        try
        {
            full = Path.TrimEndingDirectorySeparator(Path.GetFullPath(directoryPath));
        }
        catch (Exception e) when (e is ArgumentException or NotSupportedException or PathTooLongException)
        {
            return false;
        }

        for (var current = new DirectoryInfo(full); current.Parent?.Parent is { } grandParent; current = current.Parent)
        {
            if (!string.Equals(current.Parent.Name, "snapshots", StringComparison.Ordinal)
                || !grandParent.Name.StartsWith("models--", StringComparison.Ordinal))
                continue;

            repoDir = grandParent.FullName;
            snapshotDir = current.FullName;
            var relative = Path.GetRelativePath(snapshotDir, full);
            subfolder = relative == "." ? null : relative.Replace(Path.DirectorySeparatorChar, '/');
            return true;
        }

        return false;
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
                Size = CacheManager.TryGetContentLength(f, out var length) ? length : 0
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
