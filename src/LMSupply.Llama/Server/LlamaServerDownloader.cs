using System.Diagnostics;
using System.IO.Compression;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.RegularExpressions;
using LMSupply.Runtime;

namespace LMSupply.Llama.Server;

/// <summary>
/// Downloads llama-server binaries from GitHub releases.
/// </summary>
public sealed class LlamaServerDownloader : IDisposable
{
    private const string GitHubApiBase = "https://api.github.com/repos/ggml-org/llama.cpp";
    private const string ReleasesUrl = $"{GitHubApiBase}/releases";

    /// <summary>
    /// The asset a versioned llama.cpp release (<c>vX.Y.Z</c>) carries to name the build it was cut
    /// from. Since 2026-08-21 llama.cpp publishes versioned releases as the non-prerelease line —
    /// which is what GitHub's <c>releases/latest</c> returns — and marks the <c>bNNNNN</c> build
    /// releases (the ones that actually carry binaries) as prereleases. This pointer is how the
    /// stable line maps back to a downloadable build.
    /// </summary>
    private const string NightlyTagAssetName = "nightly-tag.txt";

    /// <summary>How many recent releases the listing fallback scans for a build release with assets.</summary>
    private const int ReleaseListingPageSize = 30;

    private static readonly Regex s_buildTagPattern = new(@"^b\d+$", RegexOptions.IgnoreCase | RegexOptions.Compiled);

    private readonly HttpClient _httpClient;
    private readonly string _cacheDirectory;
    private readonly bool _ownsHttpClient;
    private readonly bool _includePrerelease;

    /// <summary>
    /// Process-wide gate that serializes CUDA-runtime provisioning. A single static gate is the
    /// canonical lazy-once-init lock: the lock-free fast path means it is only contended during the
    /// one-time provisioning window, and concurrent callers for the same versionDir (e.g. Filer
    /// loading embedder + generator via Task.WhenAll) must converge to a single download/extract.
    /// Per-key (one semaphore per versionDir) would only add parallelism for the rare cuda12-AND-cuda13
    /// first-provision-at-once case — gold-plating until observed.
    /// </summary>
    private static readonly SemaphoreSlim s_cudartGate = new(1, 1);

    /// <summary>
    /// Creates a new downloader instance.
    /// </summary>
    /// <param name="cacheDirectory">Directory to store downloaded binaries.</param>
    /// <param name="httpClient">Optional HTTP client (creates new if null).</param>
    /// <param name="includePrerelease">
    /// When true, "latest" means the newest <c>bNNNNN</c> build release regardless of its prerelease
    /// flag (llama.cpp's nightly line). When false (default), "latest" follows the versioned stable
    /// line and resolves it to the build it names — see <see cref="GetLatestVersionAsync"/>.
    /// </param>
    public LlamaServerDownloader(string? cacheDirectory = null, HttpClient? httpClient = null, bool includePrerelease = false)
    {
        _cacheDirectory = cacheDirectory ?? LMSupplyCachePaths.GetLlamaServerDirectory();
        _includePrerelease = includePrerelease;

        if (httpClient != null)
        {
            _httpClient = httpClient;
            _ownsHttpClient = false;
        }
        else
        {
            _httpClient = new HttpClient();
            _httpClient.DefaultRequestHeaders.Add("User-Agent", "LMSupply");
            _ownsHttpClient = true;
        }
    }

    /// <summary>
    /// Gets the latest llama-server <b>build tag</b> (<c>bNNNNN</c>) — never a versioned release tag.
    /// Everything downstream (asset names, the cache layout, the state file,
    /// <see cref="LlamaServerVersionRequirements.ParseBuildNumber"/>) is keyed on build tags, so this
    /// is the single place upstream's release scheme is normalized:
    /// <list type="number">
    /// <item>With prereleases opted in, the newest build release that carries assets wins outright.</item>
    /// <item>Otherwise <c>releases/latest</c> is read. If it is itself a build release (the pre-2026-08
    /// scheme) its tag is returned as-is; if it is a versioned release its <c>nightly-tag.txt</c>
    /// asset is read and the build it names is returned.</item>
    /// <item>If neither yields a build tag, the release listing is scanned for the newest build release
    /// with assets — so a missing or malformed pointer degrades to "newest downloadable" rather than
    /// to "nothing downloadable".</item>
    /// </list>
    /// Returns null only when GitHub cannot be reached or no build release with assets exists.
    /// </summary>
    public async Task<string?> GetLatestVersionAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            if (_includePrerelease)
                return await FindNewestBuildReleaseAsync(cancellationToken);

            string? build;
            using (var latest = await GetJsonAsync($"{ReleasesUrl}/latest", cancellationToken))
            {
                build = await ResolveBuildTagAsync(latest.RootElement, cancellationToken);
            }

            return build ?? await FindNewestBuildReleaseAsync(cancellationToken);
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[LlamaServerDownloader] Latest release fetch failed: {ex.Message}");
            return null;
        }
    }

    /// <summary>True for a llama.cpp build tag such as <c>b10809</c>.</summary>
    internal static bool IsBuildTag(string? tag) => tag != null && s_buildTagPattern.IsMatch(tag);

    /// <summary>
    /// Maps one GitHub release payload to the build tag it stands for: a build release maps to its own
    /// tag; a versioned release maps to the build named by its <see cref="NightlyTagAssetName"/> asset.
    /// Null when the payload names neither (no pointer asset, or a pointer that is not a build tag).
    /// </summary>
    private async Task<string?> ResolveBuildTagAsync(JsonElement release, CancellationToken cancellationToken)
    {
        var tag = release.GetProperty("tag_name").GetString();
        if (IsBuildTag(tag))
            return tag;

        if (!release.TryGetProperty("assets", out var assets))
            return null;

        foreach (var asset in assets.EnumerateArray())
        {
            if (!string.Equals(asset.GetProperty("name").GetString(), NightlyTagAssetName, StringComparison.OrdinalIgnoreCase))
                continue;

            var url = asset.GetProperty("browser_download_url").GetString();
            if (url == null)
                return null;

            var pointer = (await _httpClient.GetStringAsync(url, cancellationToken)).Trim();
            if (IsBuildTag(pointer))
                return pointer;

            Trace.TraceInformation(
                $"[LlamaServerDownloader] Release '{tag}' has a {NightlyTagAssetName} that does not name a build tag: '{pointer}'.");
            return null;
        }

        return null;
    }

    /// <summary>
    /// Scans the most recent releases for the highest-numbered build release that actually carries
    /// assets, prerelease or not. Null when the page holds none.
    /// </summary>
    private async Task<string?> FindNewestBuildReleaseAsync(CancellationToken cancellationToken)
    {
        using var listing = await GetJsonAsync($"{ReleasesUrl}?per_page={ReleaseListingPageSize}", cancellationToken);

        string? newest = null;
        var newestBuild = -1;
        foreach (var release in listing.RootElement.EnumerateArray())
        {
            var tag = release.GetProperty("tag_name").GetString();
            if (!IsBuildTag(tag))
                continue;
            if (!release.TryGetProperty("assets", out var assets) || assets.GetArrayLength() == 0)
                continue;

            var build = LlamaServerVersionRequirements.ParseBuildNumber(tag) ?? -1;
            if (build > newestBuild)
            {
                newestBuild = build;
                newest = tag;
            }
        }

        return newest;
    }

    private async Task<JsonDocument> GetJsonAsync(string url, CancellationToken cancellationToken)
    {
        using var response = await _httpClient.GetAsync(url, cancellationToken);
        response.EnsureSuccessStatusCode();
        return await JsonDocument.ParseAsync(
            await response.Content.ReadAsStreamAsync(cancellationToken),
            cancellationToken: cancellationToken);
    }

    /// <summary>
    /// Gets the available asset for the current platform and preferred backend.
    /// <paramref name="version"/> may be a build tag (<c>b10809</c>) or a versioned release tag
    /// (<c>v0.4.0</c>); the latter is normalized to the build it names, so the returned asset's
    /// <see cref="LlamaServerAsset.Version"/> — and therefore the cache directory — is always a build tag.
    /// </summary>
    public async Task<LlamaServerAsset?> GetAssetAsync(
        string? version = null,
        LlamaServerBackend? preferredBackend = null,
        CancellationToken cancellationToken = default)
        => (await ResolveAssetAsync(version, preferredBackend, cancellationToken)).Asset;

    /// <summary>
    /// Resolves the asset to download and, when there is none, keeps enough of the search to explain why.
    /// </summary>
    /// <remarks>
    /// Acquisition can come up empty for three unrelated reasons — no release resolved, the tag did not
    /// resolve to a build, or the release simply has no asset for this platform — and until 0.61.0 all three
    /// surfaced as one sentence naming a backend. That was actively misleading: the backend it named was the
    /// last link of the CPU fallback chain rather than the one requested, so a consumer on a GPU-less machine
    /// read an upstream tag-resolution failure as "there is no CPU build" and went looking for a gap that does
    /// not exist. The caller throws; this only makes sure the throw can say something true.
    /// </remarks>
    public async Task<LlamaServerAssetResolution> ResolveAssetAsync(
        string? version = null,
        LlamaServerBackend? preferredBackend = null,
        CancellationToken cancellationToken = default)
    {
        var platform = GetCurrentPlatform();
        var arch = GetCurrentArchitecture();
        var requested = preferredBackend ?? GetPreferredBackend(platform);

        var requestedVersion = version;
        version ??= await GetLatestVersionAsync(cancellationToken);
        if (version == null)
        {
            return LlamaServerAssetResolution.Failed(
                platform, arch, requested,
                releaseTag: null,
                reason: LlamaServerAcquisitionFailure.ReleaseNotResolved,
                failure: requestedVersion is null
                    ? "no llama.cpp release could be resolved (the latest-release lookup returned nothing)"
                    : $"release '{requestedVersion}' could not be resolved");
        }

        if (!IsBuildTag(version))
        {
            using var release = await GetJsonAsync($"{ReleasesUrl}/tags/{version}", cancellationToken);
            var resolved = await ResolveBuildTagAsync(release.RootElement, cancellationToken);
            if (resolved == null)
            {
                return LlamaServerAssetResolution.Failed(
                    platform, arch, requested,
                    releaseTag: version,
                    reason: LlamaServerAcquisitionFailure.ReleaseTagNotABuild,
                    failure: $"release '{version}' does not name a build tag (no nightly-tag asset, and no build " +
                             "release could be found)");
            }

            version = resolved;
        }

        using var doc = await GetJsonAsync($"{ReleasesUrl}/tags/{version}", cancellationToken);
        var assets = doc.RootElement.GetProperty("assets");

        var available = new List<string>();
        foreach (var asset in assets.EnumerateArray())
        {
            if (asset.GetProperty("name").GetString() is { } name)
                available.Add(name);
        }

        // The CPU fallback is walked here rather than by recursion so the chain that was actually tried can be
        // reported. Recursing lost it: the failure then named whatever backend the innermost call held.
        var chain = requested == LlamaServerBackend.Cpu
            ? new[] { requested }
            : [requested, LlamaServerBackend.Cpu];

        foreach (var backend in chain)
        {
            var assetPattern = GetAssetPattern(platform, arch, backend);

            foreach (var asset in assets.EnumerateArray())
            {
                var name = asset.GetProperty("name").GetString();
                if (name != null && assetPattern.IsMatch(name))
                {
                    return LlamaServerAssetResolution.Found(new LlamaServerAsset
                    {
                        Name = name,
                        DownloadUrl = asset.GetProperty("browser_download_url").GetString()!,
                        Version = version,
                        Platform = platform,
                        Backend = backend,
                        Architecture = arch,
                        SizeBytes = asset.TryGetProperty("size", out var size) ? size.GetInt64() : null
                    });
                }
            }
        }

        return LlamaServerAssetResolution.Failed(
            platform, arch, requested,
            releaseTag: version,
            reason: LlamaServerAcquisitionFailure.NoAssetForPlatform,
            failure: "no asset in that release matches this platform",
            backendsTried: chain,
            availableAssets: available);
    }

    /// <summary>
    /// Downloads and extracts llama-server to the cache directory.
    /// Returns the path to the llama-server executable.
    /// </summary>
    public async Task<string> DownloadAsync(
        LlamaServerAsset asset,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var versionDir = Path.Combine(_cacheDirectory, asset.Version, asset.Backend.ToString().ToLowerInvariant());
        var serverPath = GetServerExecutablePath(versionDir, asset.Platform);

        // Check if already downloaded
        if (File.Exists(serverPath))
        {
            return serverPath;
        }

        Directory.CreateDirectory(versionDir);

        // Download archive
        var archivePath = Path.Combine(versionDir, asset.Name);

        progress?.Report(new DownloadProgress
        {
            FileName = asset.Name,
            BytesDownloaded = 0,
            TotalBytes = asset.SizeBytes ?? 0,
            Phase = DownloadPhase.Downloading
        });

        using (var response = await _httpClient.GetAsync(asset.DownloadUrl, HttpCompletionOption.ResponseHeadersRead, cancellationToken))
        {
            response.EnsureSuccessStatusCode();

            var totalBytes = response.Content.Headers.ContentLength ?? asset.SizeBytes ?? 0;
            var tracker = new DownloadProgressTracker();
            tracker.Start();

            await using var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken);
            await using var fileStream = File.Create(archivePath);

            var buffer = new byte[81920];
            long bytesDownloaded = 0;
            int bytesRead;

            while ((bytesRead = await contentStream.ReadAsync(buffer, cancellationToken)) > 0)
            {
                await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), cancellationToken);
                bytesDownloaded += bytesRead;

                progress?.Report(tracker.CreateProgress(asset.Name, bytesDownloaded, totalBytes, DownloadPhase.Downloading));
            }
        }

        // Extract archive
        progress?.Report(new DownloadProgress
        {
            FileName = asset.Name,
            BytesDownloaded = 0,
            TotalBytes = 0,
            Phase = DownloadPhase.Extracting
        });

        await ExtractArchiveAsync(archivePath, versionDir, asset.Platform, cancellationToken);

        // A download + extract that completed without throwing does NOT guarantee the expected
        // executable is actually there -- an archive layout mismatch (see
        // FlattenSingleTopLevelDirectory) or a silently incomplete extraction would otherwise return
        // a path nothing launches, and the real cause would only surface later as an opaque
        // process-start failure far from here (2026-08-17: exactly what happened on the Linux e2e
        // runner before this check existed -- confirmed against the actual b10290 release asset).
        if (!File.Exists(serverPath))
        {
            var extracted = Directory.Exists(versionDir)
                ? string.Join(", ", Directory.EnumerateFileSystemEntries(versionDir).Select(Path.GetFileName))
                : "(directory does not exist)";
            throw new InvalidOperationException(
                $"llama-server binary not found at '{serverPath}' after downloading and extracting " +
                $"'{asset.Name}'. The archive was downloaded successfully but extraction did not " +
                $"produce the expected executable -- the release's archive layout may have changed. " +
                $"Extracted contents of '{versionDir}': {extracted}.");
        }

        progress?.Report(new DownloadProgress
        {
            FileName = asset.Name,
            BytesDownloaded = 100,
            TotalBytes = 100,
            Phase = DownloadPhase.Complete
        });

        return serverPath;
    }

    /// <summary>
    /// Ensures llama-server is available, downloading if necessary.
    /// </summary>
    public async Task<string> EnsureServerAsync(
        string? version = null,
        LlamaServerBackend? preferredBackend = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var resolution = await ResolveAssetAsync(version, preferredBackend, cancellationToken);
        if (resolution.Asset == null)
            throw new LlamaServerAcquisitionException(resolution);

        return await DownloadAsync(resolution.Asset, progress, cancellationToken);
    }

    /// <summary>
    /// Gets the path to a cached llama-server version, or null if not cached.
    /// </summary>
    public string? GetCachedServerPath(string version, LlamaServerBackend backend)
    {
        var versionDir = Path.Combine(_cacheDirectory, version, backend.ToString().ToLowerInvariant());
        var serverPath = GetServerExecutablePath(versionDir, GetCurrentPlatform());

        return File.Exists(serverPath) ? serverPath : null;
    }

    /// <summary>
    /// Gets all cached build versions, newest build first. Ordered by build <i>number</i>, not by
    /// string — lexically "b9999" sorts after "b10809", which would have made a stale cached build
    /// win over a newer one once builds crossed b10000.
    /// </summary>
    public IReadOnlyList<string> GetCachedVersions()
    {
        if (!Directory.Exists(_cacheDirectory))
            return Array.Empty<string>();

        return Directory.GetDirectories(_cacheDirectory)
            .Select(Path.GetFileName)
            .Where(IsBuildTag)
            .OrderByDescending(v => LlamaServerVersionRequirements.ParseBuildNumber(v))
            .ToList()!;
    }

    private static string GetServerExecutablePath(string directory, LlamaServerPlatform platform)
    {
        var executable = platform == LlamaServerPlatform.Windows ? "llama-server.exe" : "llama-server";
        return Path.Combine(directory, executable);
    }

    private static async Task ExtractArchiveAsync(
        string archivePath,
        string destinationDir,
        LlamaServerPlatform platform,
        CancellationToken cancellationToken)
    {
        if (archivePath.EndsWith(".zip", StringComparison.OrdinalIgnoreCase))
        {
            await Task.Run(() => ZipFile.ExtractToDirectory(archivePath, destinationDir, overwriteFiles: true), cancellationToken);
        }
        else if (archivePath.EndsWith(".tar.gz", StringComparison.OrdinalIgnoreCase))
        {
            await ExtractTarGzAsync(archivePath, destinationDir, cancellationToken);
        }

        // Must happen before FlattenSingleTopLevelDirectory: the archive was downloaded directly into
        // destinationDir (see DownloadAsync), so until it is removed it sits alongside the extracted
        // tree as a spurious second top-level entry and defeats the single-wrapper-directory check.
        File.Delete(archivePath);

        FlattenSingleTopLevelDirectory(destinationDir);

        // Set executable permission on Unix
        if (platform != LlamaServerPlatform.Windows)
        {
            var serverPath = GetServerExecutablePath(destinationDir, platform);
            if (File.Exists(serverPath) && !OperatingSystem.IsWindows())
            {
                // chmod +x
                File.SetUnixFileMode(serverPath,
                    UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute |
                    UnixFileMode.GroupRead | UnixFileMode.GroupExecute |
                    UnixFileMode.OtherRead | UnixFileMode.OtherExecute);
            }
        }
    }

    /// <summary>
    /// Flattens a single top-level wrapper directory left by extraction. llama.cpp's Linux/macOS
    /// tar.gz releases unpack as "llama-b10290/llama-server" rather than "llama-server" at the
    /// archive root (confirmed 2026-08-17 against the actual b10290 ubuntu-x64 asset); the Windows
    /// zip for the same release is already flat, which is why the resulting missing-binary failure
    /// only ever reproduced on Linux/macOS runners. Every other code path in this file
    /// (<see cref="GetServerExecutablePath"/>, <see cref="GetCachedServerPath"/>,
    /// <see cref="CudaRuntimePresent"/>) assumes a flat <paramref name="destinationDir"/>, so
    /// normalizing once here keeps that assumption true everywhere instead of teaching every caller
    /// about archive layout. No-op when the archive was already flat (zero or multiple top-level
    /// entries, or a single top-level file rather than a directory).
    /// </summary>
    internal static void FlattenSingleTopLevelDirectory(string destinationDir)
    {
        var entries = Directory.GetFileSystemEntries(destinationDir);
        if (entries.Length != 1 || !Directory.Exists(entries[0]))
            return;

        var wrapperDir = entries[0];
        foreach (var entry in Directory.GetFileSystemEntries(wrapperDir))
        {
            var target = Path.Combine(destinationDir, Path.GetFileName(entry));
            if (Directory.Exists(entry))
                Directory.Move(entry, target);
            else
                File.Move(entry, target);
        }

        Directory.Delete(wrapperDir);
    }

    private static async Task ExtractTarGzAsync(string archivePath, string destinationDir, CancellationToken cancellationToken)
    {
        await using var fileStream = File.OpenRead(archivePath);
        await using var gzipStream = new GZipStream(fileStream, CompressionMode.Decompress);
        await System.Formats.Tar.TarFile.ExtractToDirectoryAsync(gzipStream, destinationDir, overwriteFiles: true, cancellationToken);
    }

    private static LlamaServerPlatform GetCurrentPlatform()
    {
        if (OperatingSystem.IsWindows()) return LlamaServerPlatform.Windows;
        if (OperatingSystem.IsMacOS()) return LlamaServerPlatform.MacOS;
        return LlamaServerPlatform.Linux;
    }

    private static LlamaServerArchitecture GetCurrentArchitecture()
    {
        return System.Runtime.InteropServices.RuntimeInformation.OSArchitecture switch
        {
            System.Runtime.InteropServices.Architecture.Arm64 => LlamaServerArchitecture.Arm64,
            _ => LlamaServerArchitecture.X64
        };
    }

    private static LlamaServerBackend GetPreferredBackend(LlamaServerPlatform platform)
    {
        // Default preferences based on platform
        return platform switch
        {
            LlamaServerPlatform.MacOS => LlamaServerBackend.Metal,
            LlamaServerPlatform.Windows => LlamaServerBackend.Vulkan, // Vulkan has good AMD/Intel/NVIDIA support
            LlamaServerPlatform.Linux => LlamaServerBackend.Vulkan,
            _ => LlamaServerBackend.Cpu
        };
    }

    private static Regex GetAssetPattern(LlamaServerPlatform platform, LlamaServerArchitecture arch, LlamaServerBackend backend)
    {
        var os = platform switch
        {
            LlamaServerPlatform.Windows => "win",
            LlamaServerPlatform.MacOS => "macos",
            LlamaServerPlatform.Linux => "ubuntu",
            _ => throw new NotSupportedException()
        };

        var archStr = arch switch
        {
            LlamaServerArchitecture.Arm64 => "arm64",
            LlamaServerArchitecture.X64 => "x64",
            _ => throw new NotSupportedException()
        };

        var backendStr = backend switch
        {
            LlamaServerBackend.Cpu => "cpu",
            LlamaServerBackend.Vulkan => "vulkan",
            LlamaServerBackend.Cuda12 => "cuda-12",
            LlamaServerBackend.Cuda13 => "cuda-13",
            LlamaServerBackend.Hip => "hip",
            LlamaServerBackend.Sycl => "sycl",
            LlamaServerBackend.Metal => "", // macOS arm64 has Metal by default
            _ => throw new NotSupportedException()
        };

        // Build pattern based on backend
        if (backend == LlamaServerBackend.Metal && platform == LlamaServerPlatform.MacOS)
        {
            // macOS arm64 Metal: llama-b7898-bin-macos-arm64.tar.gz
            return new Regex($@"llama-b\d+-bin-{os}-{archStr}\.(zip|tar\.gz)$", RegexOptions.IgnoreCase);
        }

        if (backend == LlamaServerBackend.Cpu)
        {
            // CPU build: llama-b7898-bin-win-cpu-x64.zip
            if (platform == LlamaServerPlatform.Windows)
                return new Regex($@"llama-b\d+-bin-{os}-cpu-{archStr}\.zip$", RegexOptions.IgnoreCase);
            // Linux CPU: llama-b7898-bin-ubuntu-x64.tar.gz (no "cpu" in name)
            return new Regex($@"llama-b\d+-bin-{os}-{archStr}\.(zip|tar\.gz)$", RegexOptions.IgnoreCase);
        }

        // GPU builds: llama-b7898-bin-win-vulkan-x64.zip
        // CUDA builds have minor version: llama-b7902-bin-win-cuda-12.4-x64.zip
        // HIP builds have suffix: llama-b7902-bin-win-hip-radeon-x64.zip
        var backendPattern = backend switch
        {
            LlamaServerBackend.Cuda12 or LlamaServerBackend.Cuda13 => $@"{backendStr}\.\d+",
            LlamaServerBackend.Hip => @"hip-radeon",
            _ => backendStr
        };
        return new Regex($@"llama-b\d+-bin-{os}-{backendPattern}-{archStr}\.(zip|tar\.gz)$", RegexOptions.IgnoreCase);
    }

    /// <summary>
    /// Builds a regex matching the CUDA-runtime companion asset (cudart/cublas/cublasLt) that
    /// llama.cpp ships SEPARATELY for a CUDA backend, e.g. "cudart-llama-bin-win-cuda-12.4-x64.zip".
    /// The main "llama-b&lt;n&gt;-bin-...-cuda-..." archive does NOT contain the runtime, so without
    /// this companion the cuda binary silently falls back to CPU. Matches any cuda minor for the
    /// backend major. Returns null for non-CUDA backends (no companion needed) and for macOS
    /// (no CUDA).
    /// </summary>
    internal static Regex? GetCudartAssetPattern(
        LlamaServerPlatform platform, LlamaServerArchitecture arch, LlamaServerBackend backend)
    {
        var major = backend switch
        {
            LlamaServerBackend.Cuda12 => "12",
            LlamaServerBackend.Cuda13 => "13",
            _ => null
        };
        if (major == null)
            return null;

        var os = platform switch
        {
            LlamaServerPlatform.Windows => "win",
            LlamaServerPlatform.Linux => "ubuntu",
            _ => null // macOS has no CUDA build
        };
        if (os == null)
            return null;

        var archStr = arch switch
        {
            LlamaServerArchitecture.Arm64 => "arm64",
            _ => "x64"
        };

        return new Regex(
            $@"^cudart-llama-bin-{os}-cuda-{major}\.\d+-{archStr}\.(zip|tar\.gz)$",
            RegexOptions.IgnoreCase);
    }

    /// <summary>
    /// True only if the COMPLETE CUDA runtime (cudart + cublas + cublasLt) is present in
    /// <paramref name="versionDir"/>. All three are required: a partial state (e.g. the small cudart
    /// present but the 473 MB cublasLt truncated/missing after an interrupted or concurrent extract)
    /// must NOT count as present, otherwise the cache is permanently poisoned — ggml-cuda fails to
    /// load, the server silently runs on CPU, and a re-run never re-provisions. Requiring all three
    /// makes a partial state self-heal on the next load. Accepts both Windows (cudart64_*.dll) and
    /// Linux (libcudart.so*) naming so the check is OS-agnostic.
    /// </summary>
    internal static bool CudaRuntimePresent(string versionDir)
    {
        if (!Directory.Exists(versionDir))
            return false;

        bool HasFamily(string winGlob, string soGlob)
            => Directory.EnumerateFiles(versionDir, winGlob).Any()
            || Directory.EnumerateFiles(versionDir, soGlob).Any();

        return HasFamily("cudart64_*.dll", "libcudart.so*")
            && HasFamily("cublas64_*.dll", "libcublas.so*")
            && HasFamily("cublasLt64_*.dll", "libcublasLt.so*");
    }

    /// <summary>
    /// Ensures the CUDA runtime (cudart/cublas/cublasLt) is present next to a CUDA llama-server
    /// binary. llama.cpp ships the runtime in a separate release asset; without it ggml-cuda fails
    /// to load and the server silently runs on CPU. Idempotent (no-op when already present or for
    /// non-CUDA backends) and best-effort: any failure is logged and swallowed so model loading is
    /// not blocked — the runtime silent-fallback guard in <see cref="LlamaServerProcess"/> will then
    /// surface a warning at startup. Covers already-cached binaries (the common case where a prior
    /// version downloaded the cuda binary before this companion logic existed).
    /// </summary>
    public async Task EnsureCudaRuntimeAsync(
        string versionDir,
        LlamaServerBackend backend,
        string version,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var pattern = GetCudartAssetPattern(GetCurrentPlatform(), GetCurrentArchitecture(), backend);
        if (pattern == null)
            return; // not a CUDA backend (or macOS): no companion runtime needed

        try
        {
            // Serialize provisioning so concurrent callers for the same versionDir (e.g. Filer loading
            // embedder + generator via Task.WhenAll) converge to a single download/extract. The gate is
            // held across the (multi-minute) download on purpose: that is what lets a waiting caller
            // find the runtime already present at the re-check and download nothing.
            await RunGatedOnceAsync(
                s_cudartGate,
                alreadyDone: () => CudaRuntimePresent(versionDir),
                work: () => ProvisionCudaRuntimeCoreAsync(
                    versionDir, backend, version, pattern, progress, cancellationToken),
                cancellationToken);
        }
        catch (Exception ex)
        {
            // Best-effort: never block model loading on companion provisioning. This also swallows
            // OperationCanceledException; RunGatedOnceAsync's acquired guard ensures the gate is never
            // corrupted by a release of a semaphore that WaitAsync did not take.
            Trace.TraceWarning(
                $"[LlamaServerDownloader] Failed to provision CUDA runtime for {backend} " +
                $"version {version}: {ex.Message}. The cuda binary may run on CPU.");
        }
    }

    /// <summary>
    /// Runs <paramref name="work"/> at most once across concurrent callers gated by
    /// <paramref name="gate"/>. Callers take a lock-free fast path when <paramref name="alreadyDone"/>
    /// is already true; otherwise they serialize on the gate and RE-CHECK after acquiring, so a caller
    /// that waited while another completed the work does nothing. The <c>acquired</c> guard ensures a
    /// wait cancelled before acquisition never releases a semaphore it did not take. Exceptions
    /// (including cancellation) propagate; callers that want best-effort semantics wrap the call.
    /// </summary>
    internal static async Task RunGatedOnceAsync(
        SemaphoreSlim gate,
        Func<bool> alreadyDone,
        Func<Task> work,
        CancellationToken cancellationToken)
    {
        if (alreadyDone())
            return;

        var acquired = false;
        try
        {
            await gate.WaitAsync(cancellationToken);
            acquired = true;

            if (alreadyDone())
                return;

            await work();
        }
        finally
        {
            if (acquired)
                gate.Release();
        }
    }

    /// <summary>
    /// Downloads and extracts the cudart companion asset for one provisioning pass. Assumes the caller
    /// has already verified this is a CUDA backend (<paramref name="pattern"/> non-null) and that the
    /// runtime is not yet present. Throws on failure; <see cref="EnsureCudaRuntimeAsync"/> swallows it.
    /// </summary>
    private async Task ProvisionCudaRuntimeCoreAsync(
        string versionDir,
        LlamaServerBackend backend,
        string version,
        Regex pattern,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        // Find the cudart companion asset in the same release.
        var releaseUrl = $"{ReleasesUrl}/tags/{version}";
        using var response = await _httpClient.GetAsync(releaseUrl, cancellationToken);
        response.EnsureSuccessStatusCode();

        using var doc = await JsonDocument.ParseAsync(
            await response.Content.ReadAsStreamAsync(cancellationToken),
            cancellationToken: cancellationToken);

        string? cudartName = null;
        string? cudartUrl = null;
        foreach (var asset in doc.RootElement.GetProperty("assets").EnumerateArray())
        {
            var name = asset.GetProperty("name").GetString();
            if (name != null && pattern.IsMatch(name))
            {
                cudartName = name;
                cudartUrl = asset.GetProperty("browser_download_url").GetString();
                break;
            }
        }

        if (cudartName == null || cudartUrl == null)
        {
            Trace.TraceWarning(
                $"[LlamaServerDownloader] CUDA runtime companion not found for {backend} " +
                $"version {version}. The cuda binary may run on CPU. Expected an asset like " +
                "cudart-llama-bin-...-cuda-XX.Y-...");
            return;
        }

        Directory.CreateDirectory(versionDir);

        // Provision atomically w.r.t. the versionDir: download + extract into an isolated staging
        // directory (kept INSIDE versionDir so it is on the same volume, making the final move an
        // atomic rename), then move only a fully extracted runtime into place. A failed or interrupted
        // extract therefore never leaves a partial state in the versionDir — the CudaRuntimePresent
        // completeness check stays meaningful and a truncated cublasLt is never load-attempted by ggml.
        // The staging directory is always removed (success or failure). A unique name keeps concurrent
        // cross-process provisioners (which the in-process gate cannot serialize) from colliding.
        var stagingDir = Path.Combine(versionDir, ".cudart-staging-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(stagingDir);
        try
        {
            var archivePath = Path.Combine(stagingDir, cudartName);

            progress?.Report(new DownloadProgress { FileName = cudartName, Phase = DownloadPhase.Downloading });

            using (var dl = await _httpClient.GetAsync(cudartUrl, HttpCompletionOption.ResponseHeadersRead, cancellationToken))
            {
                dl.EnsureSuccessStatusCode();
                await using var src = await dl.Content.ReadAsStreamAsync(cancellationToken);
                await using var dst = File.Create(archivePath);
                await src.CopyToAsync(dst, cancellationToken);
            }

            var extractDir = Path.Combine(stagingDir, "extracted");
            Directory.CreateDirectory(extractDir);
            await ExtractArchiveAsync(archivePath, extractDir, GetCurrentPlatform(), cancellationToken);

            // Move the extracted runtime into the versionDir. cublasLt (the largest file and the last
            // family CudaRuntimePresent requires) is moved LAST so the completeness check only flips
            // true once every family is in place — a crash mid-move leaves the versionDir incomplete
            // (present=false) and the next load re-provisions rather than running on a partial runtime.
            // The cudart companion archive is flat (runtime DLLs at the root), so flattening via
            // Path.GetFileName is safe; AllDirectories is a defensive net for any incidental subdir.
            foreach (var file in Directory.GetFiles(extractDir, "*", SearchOption.AllDirectories)
                         .OrderBy(CudartMoveOrder))
            {
                File.Move(file, Path.Combine(versionDir, Path.GetFileName(file)), overwrite: true);
            }
        }
        finally
        {
            try { Directory.Delete(stagingDir, recursive: true); }
            catch { /* best-effort staging cleanup; the runtime is already in place */ }
        }

        Trace.TraceInformation(
            $"[LlamaServerDownloader] Installed CUDA runtime {cudartName} for {backend} into {versionDir}.");
    }

    /// <summary>Sort key that moves the cublasLt family last (see <see cref="ProvisionCudaRuntimeCoreAsync"/>).</summary>
    internal static int CudartMoveOrder(string path)
    {
        var name = Path.GetFileName(path);
        return name.StartsWith("cublasLt", StringComparison.OrdinalIgnoreCase)
            || name.StartsWith("libcublasLt", StringComparison.OrdinalIgnoreCase) ? 1 : 0;
    }

    public void Dispose()
    {
        if (_ownsHttpClient)
        {
            _httpClient.Dispose();
        }
    }
}
