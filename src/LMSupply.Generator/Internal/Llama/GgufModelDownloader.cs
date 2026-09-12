using System.Diagnostics;
using LMSupply;
using LMSupply.Core.Download;
using LMSupply.Hardware;
using LMSupply.Download;
using LMSupply.Exceptions;

namespace LMSupply.Generator.Internal.Llama;

/// <summary>
/// Downloads GGUF model files from HuggingFace with automatic quantization selection.
/// </summary>
public sealed class GgufModelDownloader : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly ModelDiscoveryService _discoveryService;
    private readonly string _cacheDirectory;
    private readonly bool _localFilesOnly;
    private bool _disposed;

    private const string HuggingFaceFileBase = "https://huggingface.co";

    /// <summary>
    /// Default quantization preference order (best balance of quality vs size first).
    /// </summary>
    private static readonly string[] DefaultQuantizationPriority =
    [
        "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S",
        "Q6_K", "Q8_0", "Q3_K_M", "Q3_K_L", "Q2_K",
        "IQ4_XS", "IQ4_NL"
    ];

    public GgufModelDownloader() : this(null)
    {
    }

    /// <param name="cacheDirectory">Where downloaded GGUF files are kept. Null uses the default.</param>
    /// <param name="hfToken">
    /// HuggingFace access token. Null falls back to the <c>HF_TOKEN</c> environment variable, the
    /// same source the ONNX download path reads.
    /// </param>
    /// <remarks>
    /// Until 2026-08-06 this path sent no credentials at all, while the sibling ONNX path
    /// (<c>ModelDiscoveryService</c>, <c>ModelMetadataService</c>) authenticated with the very same
    /// <c>HF_TOKEN</c> — and the error text there tells the caller to set it. Following that advice
    /// did nothing for GGUF downloads, which are the large ones: unauthenticated requests share a
    /// per-IP rate limit, so a cold pull on a shared CI runner gets 429, and a failed download never
    /// creates the cache directory, so nothing is cached and the next run starts in the same place.
    ///
    /// Deliberately NOT copied from the sibling: its 30-second timeout and automatic decompression.
    /// Those are tuned for small JSON responses; this client streams multi-gigabyte files.
    /// </remarks>
    /// <param name="localFilesOnly">
    /// True: serve from the cache only — no repository listing, no download; a file that is not cached fails
    /// with <see cref="ModelNotFoundException"/>. <c>GeneratorOptions.DisableAutoDownload</c> maps here.
    /// </param>
    public GgufModelDownloader(string? cacheDirectory, string? hfToken = null, bool localFilesOnly = false)
    {
        _cacheDirectory = cacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        _localFilesOnly = localFilesOnly;
        _httpClient = new HttpClient
        {
            Timeout = TimeSpan.FromMinutes(30)
        };
        _httpClient.DefaultRequestHeaders.Add("User-Agent", "LMSupply/1.0");

        var token = hfToken ?? Environment.GetEnvironmentVariable("HF_TOKEN");
        if (!string.IsNullOrEmpty(token))
        {
            _httpClient.DefaultRequestHeaders.Authorization =
                new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", token);
        }

        _discoveryService = new ModelDiscoveryService(_cacheDirectory, hfToken);
    }

    /// <summary>
    /// The Authorization scheme this downloader will send, or null when it sends none.
    /// Exists so a test can observe that credentials are actually attached — the defect this
    /// replaces was invisible from the outside: requests simply went out unauthenticated and
    /// succeeded until a rate limit was reached.
    /// </summary>
    internal string? AuthorizationScheme => _httpClient.DefaultRequestHeaders.Authorization?.Scheme;

    /// <summary>
    /// Downloads a GGUF model file from HuggingFace.
    /// </summary>
    /// <param name="repoId">HuggingFace repository ID (e.g., "bartowski/Llama-3.2-3B-Instruct-GGUF").</param>
    /// <param name="filename">Specific file to download. If null, auto-selects based on quantization.</param>
    /// <param name="preferredQuantization">Preferred quantization (e.g., "Q4_K_M"). If null, uses default priority.</param>
    /// <param name="progress">Optional progress reporter.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Path to the downloaded GGUF file.</returns>
    public async Task<string> DownloadAsync(
        string repoId,
        string? filename = null,
        string? preferredQuantization = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        // Determine the file to download — offline, only a cached file can be chosen.
        if (string.IsNullOrEmpty(filename))
        {
            filename = _localFilesOnly
                ? TrySelectFromLocalCache(repoId, preferredQuantization)
                    ?? throw NotCached(repoId, preferredQuantization is null ? "*.gguf" : $"*{preferredQuantization}*.gguf")
                : await SelectBestGgufFileAsync(repoId, preferredQuantization, cancellationToken);
        }

        // Check cache
        var cachedPath = GetCachedPath(repoId, filename);
        if (File.Exists(cachedPath))
        {
            progress?.Report(new DownloadProgress
            {
                FileName = filename,
                BytesDownloaded = 1,
                TotalBytes = 1
            });
            return cachedPath;
        }

        if (_localFilesOnly)
            throw NotCached(repoId, filename);

        // Download the file
        progress?.Report(new DownloadProgress
        {
            FileName = filename,
            BytesDownloaded = 0,
            TotalBytes = 0
        });

        await DownloadFileAsync(repoId, filename, cachedPath, progress, cancellationToken);

        return cachedPath;
    }

    /// <summary>
    /// Downloads a model using registry information.
    /// </summary>
    public async Task<string> DownloadFromRegistryAsync(
        GgufModelInfo modelInfo,
        ExecutionProvider provider = ExecutionProvider.Auto,
        string? preferredQuantization = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // Use registry default file unless a different quantization is explicitly preferred.
        var filename = modelInfo.DefaultFile;

        if (!string.IsNullOrEmpty(preferredQuantization) &&
            !modelInfo.DefaultFile.Contains(preferredQuantization, StringComparison.OrdinalIgnoreCase))
        {
            // Explicit quantization request: try to find that file — in the cache only when offline.
            var alternateFile = _localFilesOnly
                ? TrySelectFromLocalCache(modelInfo.RepoId, preferredQuantization)
                : await TryFindQuantizedFileAsync(modelInfo.RepoId, preferredQuantization, cancellationToken);

            if (alternateFile != null)
            {
                filename = alternateFile;
            }
        }
        else if (string.IsNullOrEmpty(preferredQuantization) && modelInfo.ShardCount is not > 1)
        {
            // Auto path (no explicit quant, single-file model): pick a quantization that fits the
            // backend-consistent memory budget — keep the registry default when it fits, otherwise
            // downscale to a smaller quant so low-spec/integrated-GPU hosts load instead of OOMing.
            filename = await SelectRegistryFileAsync(modelInfo, provider, cancellationToken);
        }

        // Handle split GGUF models (multiple shards) — downscaling not applicable to shards.
        if (modelInfo.ShardCount is > 1)
        {
            return await DownloadSplitModelAsync(
                modelInfo.RepoId, filename, modelInfo.ShardCount.Value,
                progress, cancellationToken);
        }

        return await DownloadAsync(modelInfo.RepoId, filename, preferredQuantization, progress, cancellationToken);
    }

    /// <summary>
    /// Auto-selects the GGUF quantization file for a registry model under a backend-consistent memory
    /// budget. Cache-first (offline-friendly); on repo-listing failure degrades to the registry default.
    /// </summary>
    private async Task<string> SelectRegistryFileAsync(
        GgufModelInfo modelInfo,
        ExecutionProvider provider,
        CancellationToken cancellationToken)
    {
        var profile = HardwareProfile.Current;
        var cpuBackend = global::LMSupply.Llama.LlamaBackendSelector.MapProvider(provider, profile.GpuInfo)
            == global::LMSupply.Llama.Server.LlamaServerBackend.Cpu;
        var budget = BuildSelectionBudget(
            cpuBackend, profile.GpuInfo, profile.SystemMemoryBytes,
            GgufModelRegistry.DefaultBudgetContextLength, out var vramOnly);

        // Offline-first, but budget-aware: only reuse a cached quant if it actually fits the budget.
        // (A cached default that no longer fits must not short-circuit the downscale.)
        var cachedGroups = ListCachedGroups(modelInfo.RepoId);
        if (cachedGroups.Count > 0)
        {
            var cachedDecision = DecideRegistryFile(modelInfo, cachedGroups, budget, vramOnly);
            if (cachedDecision.Reason is RegistryFileReason.DefaultFits or RegistryFileReason.Downscaled)
                return cachedDecision.FileName;
        }

        // Offline: whatever is cached is the choice (DownloadAsync fails the load if nothing is); never list.
        if (_localFilesOnly)
        {
            return cachedGroups.Count > 0
                ? DecideRegistryFile(modelInfo, cachedGroups, budget, vramOnly).FileName
                : modelInfo.DefaultFile;
        }

        IReadOnlyList<GgufFileGroup> groups;
        try
        {
            groups = await ListGgufGroupsAsync(modelInfo.RepoId, cancellationToken);
        }
        catch (Exception ex)
        {
            Trace.TraceWarning(
                $"[GgufModelDownloader] Repo listing failed for {modelInfo.RepoId} ({ex.Message}); " +
                $"using registry default '{modelInfo.DefaultFile}'.");
            return modelInfo.DefaultFile;
        }

        var decision = DecideRegistryFile(modelInfo, groups, budget, vramOnly);
        switch (decision.Reason)
        {
            case RegistryFileReason.FallbackSmallest:
                Trace.TraceWarning(
                    $"[GgufModelDownloader] No quantization of '{modelInfo.AliasName}' fits the memory budget; " +
                    $"using smallest '{decision.FileName}' (OOM risk). Free memory or choose a smaller model.");
                break;
            case RegistryFileReason.Downscaled:
                Trace.TraceInformation(
                    $"[GgufModelDownloader] Downscaled '{modelInfo.AliasName}' quantization to fit the memory budget: " +
                    $"'{decision.FileName}' (registry default '{modelInfo.DefaultFile}' did not fit).");
                break;
        }

        return decision.FileName;
    }

    /// <summary>
    /// Downloads all shards of a split GGUF model.
    /// Returns the path to the first shard (llama-server auto-loads the rest).
    /// </summary>
    private async Task<string> DownloadSplitModelAsync(
        string repoId,
        string firstShardFilename,
        int shardCount,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var shardFilenames = GenerateShardFilenames(firstShardFilename, shardCount);
        string? firstShardPath = null;

        for (int i = 0; i < shardFilenames.Count; i++)
        {
            var shardFile = shardFilenames[i];
            progress?.Report(new DownloadProgress
            {
                FileName = $"{Path.GetFileName(shardFile)} ({i + 1}/{shardCount})",
                BytesDownloaded = 0,
                TotalBytes = 0
            });

            var path = await DownloadAsync(repoId, shardFile, preferredQuantization: null,
                progress, cancellationToken);

            firstShardPath ??= path;
        }

        return firstShardPath!;
    }

    /// <summary>
    /// Generates all shard filenames from the first shard filename.
    /// E.g., "Q4_K_M/model-00001-of-00003.gguf" → ["...00001...", "...00002...", "...00003..."]
    /// </summary>
    internal static IReadOnlyList<string> GenerateShardFilenames(string firstShardFilename, int shardCount)
    {
        var filenames = new List<string>(shardCount);

        // Find the shard number pattern in the filename
        var match = System.Text.RegularExpressions.Regex.Match(
            firstShardFilename, @"-(\d{5})-of-(\d{5})\.gguf$",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase);

        if (!match.Success)
        {
            // Not a split pattern — return as single file
            return [firstShardFilename];
        }

        var prefix = firstShardFilename[..match.Index];
        var totalStr = match.Groups[2].Value;

        for (int i = 1; i <= shardCount; i++)
        {
            filenames.Add($"{prefix}-{i:D5}-of-{totalStr}.gguf");
        }

        return filenames;
    }

    /// <summary>
    /// Lists available GGUF files in a repository.
    /// </summary>
    public async Task<IReadOnlyList<GgufFileInfo>> ListGgufFilesAsync(
        string repoId,
        CancellationToken cancellationToken = default)
    {
        var groups = await ListGgufGroupsAsync(repoId, cancellationToken);

        return groups.Select(g => new GgufFileInfo
        {
            FileName = g.PrimaryFileName,
            Path = g.PrimaryFileName,
            SizeBytes = g.TotalSizeBytes,
            Quantization = null
        }).ToList();
    }

    /// <summary>
    /// Lists available GGUF files in a repository, grouped for split-file support.
    /// </summary>
    public async Task<IReadOnlyList<GgufFileGroup>> ListGgufGroupsAsync(
        string repoId,
        CancellationToken cancellationToken = default)
    {
        var files = await ListRepositoryFilesAsync(repoId, cancellationToken);

        var rawFiles = files
            .Where(f => f.Path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            .Where(f => !IsCompanionFile(Path.GetFileName(f.Path)))
            .Select(f => new GgufRawFile(Path.GetFileName(f.Path), f.Size))
            .ToList();

        return GgufFileGroup.GroupFiles(rawFiles).ToList();
    }

    /// <summary>Why a particular registry file/quantization was chosen by <see cref="DecideRegistryFile"/>.</summary>
    internal enum RegistryFileReason
    {
        /// <summary>The registry's default quantization fits the budget — used unchanged (capable host).</summary>
        DefaultFits,
        /// <summary>Default did not fit; a smaller quantization that fits was chosen (low-end downscale).</summary>
        Downscaled,
        /// <summary>No quantization fits the budget; the smallest was returned as best-effort (OOM risk).</summary>
        FallbackSmallest,
        /// <summary>Repository file listing was unavailable; degraded to the registry default file.</summary>
        GroupsUnavailable,
    }

    /// <summary>Result of <see cref="DecideRegistryFile"/>: the chosen file name and the reason.</summary>
    internal readonly record struct RegistryFileDecision(string FileName, RegistryFileReason Reason);

    /// <summary>
    /// Builds the memory budget for quant file selection consistent with the resolved llama.cpp backend.
    /// On a CPU backend (including integrated-GPU demotion) VRAM is zeroed so an unreliable iGPU VRAM
    /// reading is never used and selection is RAM-driven; on a GPU backend the GPU VRAM is used and
    /// <paramref name="vramOnly"/> is set so a variant is not chosen merely because system RAM is large.
    /// </summary>
    internal static AvailableMemory BuildSelectionBudget(
        bool cpuBackend,
        LMSupply.Runtime.GpuInfo gpu,
        long systemRamBytes,
        int contextLength,
        out bool vramOnly)
    {
        if (cpuBackend)
        {
            vramOnly = false;
            return new AvailableMemory(VramBytes: 0, RamBytes: systemRamBytes, contextLength);
        }

        vramOnly = true;
        return new AvailableMemory(
            VramBytes: gpu.EffectiveAvailableBytes ?? 0,
            RamBytes: systemRamBytes,
            contextLength);
    }

    /// <summary>
    /// Pure, HW/network-free decision: picks which GGUF quantization file to download for a registry
    /// model under a memory budget. Keeps the registry default quant when it fits (capable host),
    /// otherwise downscales to the largest smaller quant that fits, otherwise returns the smallest.
    /// <paramref name="vramOnly"/> selects the VRAM fit predicate (GPU backend) vs the RAM/total
    /// predicate (CPU backend) — callers pass <c>false</c> when the resolved backend is CPU so a
    /// large-but-irrelevant integrated-GPU VRAM reading is not used.
    /// </summary>
    internal static RegistryFileDecision DecideRegistryFile(
        GgufModelInfo model,
        IReadOnlyList<GgufFileGroup> availableGroups,
        AvailableMemory budget,
        bool vramOnly)
    {
        if (availableGroups.Count == 0)
            return new RegistryFileDecision(model.DefaultFile, RegistryFileReason.GroupsUnavailable);

        bool Fits(long sizeBytes) => vramOnly && budget.VramBytes > 0
            ? budget.FitsInGpu(sizeBytes)
            : budget.FitsInMemory(sizeBytes);

        // Prefer the registry's intended quant if it fits — capable hosts stay on the default.
        var registryQuant = ExtractQuantization(model.DefaultFile);
        var defaultGroup =
            availableGroups.FirstOrDefault(g =>
                g.PrimaryFileName.Equals(model.DefaultFile, StringComparison.OrdinalIgnoreCase))
            ?? (registryQuant is not null
                ? availableGroups.FirstOrDefault(g =>
                    GgufFileSelector.MatchesQuantization(g.PrimaryFileName, registryQuant))
                : null);

        if (defaultGroup is not null && Fits(defaultGroup.TotalSizeBytes))
            return new RegistryFileDecision(defaultGroup.PrimaryFileName, RegistryFileReason.DefaultFits);

        // Default doesn't fit → largest quant that fits (downscale to a smaller quant).
        var largestFitting = availableGroups
            .Where(g => Fits(g.TotalSizeBytes))
            .OrderByDescending(g => g.TotalSizeBytes)
            .FirstOrDefault();
        if (largestFitting is not null)
            return new RegistryFileDecision(largestFitting.PrimaryFileName, RegistryFileReason.Downscaled);

        // Nothing fits → smallest as best-effort (caller warns about OOM risk).
        var smallest = availableGroups.OrderBy(g => g.TotalSizeBytes).First();
        return new RegistryFileDecision(smallest.PrimaryFileName, RegistryFileReason.FallbackSmallest);
    }

    /// <summary>
    /// Selects the best GGUF file based on hardware memory constraints and quantization preference.
    /// </summary>
    private async Task<string> SelectBestGgufFileAsync(
        string repoId,
        string? preferredQuantization,
        CancellationToken cancellationToken)
    {
        // Prefer local cache over network: avoids HF API calls when the model is already downloaded.
        // Filer runs air-gapped after initial download; a live API call at boot is a correctness risk.
        var localFile = TrySelectFromLocalCache(repoId, preferredQuantization);
        if (localFile != null)
            return localFile;

        var groups = await ListGgufGroupsAsync(repoId, cancellationToken);

        if (groups.Count == 0)
            throw new ModelNotFoundException(
                $"No GGUF files found in repository '{repoId}'.",
                repoId);

        var profile = HardwareProfile.Current;
        // Backend-consistent budget: when the resolved llama.cpp backend is CPU (incl. integrated-GPU
        // demotion), VRAM is zeroed so an unreliable iGPU VRAM reading is not used (RAM-driven). On a
        // GPU backend, the VRAM budget is enforced (vramOnly) so a variant is not picked merely because
        // system RAM is large. Raw repo-ID path runs under Auto provider.
        var cpuBackend = global::LMSupply.Llama.LlamaBackendSelector.MapProvider(
                ExecutionProvider.Auto, profile.GpuInfo)
            == global::LMSupply.Llama.Server.LlamaServerBackend.Cpu;
        var memory = BuildSelectionBudget(
            cpuBackend, profile.GpuInfo, profile.SystemMemoryBytes,
            GgufModelRegistry.DefaultBudgetContextLength, out var vramOnly);
        var selected = GgufFileSelector.Select(groups, memory, preferredQuantization, vramOnly);
        return selected.PrimaryFileName;
    }

    /// <summary>
    /// Enumerates GGUF quantization variants already present in the local cache for a repo, grouped
    /// (split-file aware) with their on-disk sizes. Returns an empty list when nothing is cached.
    /// Used to reuse a cached quant that fits the budget without a network call.
    /// </summary>
    private List<GgufFileGroup> ListCachedGroups(string repoId)
    {
        var cacheDir = Path.GetDirectoryName(GetCachedPath(repoId, "placeholder.gguf"));
        if (cacheDir == null || !Directory.Exists(cacheDir))
            return [];

        var rawFiles = Directory.EnumerateFiles(cacheDir, "*.gguf", SearchOption.TopDirectoryOnly)
            .Select(p => new GgufRawFile(Path.GetFileName(p), new FileInfo(p).Length))
            .Where(f => !IsCompanionFile(f.FileName))
            .ToList();

        return rawFiles.Count == 0 ? [] : GgufFileGroup.GroupFiles(rawFiles).ToList();
    }

    /// <summary>
    /// Tries to find a file with specific quantization, checking local cache before the HF API.
    /// </summary>
    private async Task<string?> TryFindQuantizedFileAsync(
        string repoId,
        string quantization,
        CancellationToken cancellationToken)
    {
        // Local cache first
        var localCacheDir = Path.GetDirectoryName(GetCachedPath(repoId, "placeholder.gguf"));
        if (localCacheDir != null && Directory.Exists(localCacheDir))
        {
            var localMatch = Directory.EnumerateFiles(localCacheDir, "*.gguf", SearchOption.TopDirectoryOnly)
                .Select(Path.GetFileName)
                .FirstOrDefault(f => f != null &&
                    f.Contains(quantization, StringComparison.OrdinalIgnoreCase) &&
                    !IsCompanionFile(f));
            if (localMatch != null)
                return localMatch;
        }

        try
        {
            var files = await ListGgufFilesAsync(repoId, cancellationToken);
            var match = files.FirstOrDefault(f =>
                f.FileName.Contains(quantization, StringComparison.OrdinalIgnoreCase));

            return match?.FileName;
        }
        catch (Exception ex)
        {
            Trace.TraceInformation($"[GgufModelDownloader] GGUF file search failed: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Scans the local cache directory for an existing GGUF file that matches the quantization preference.
    /// Returns the filename (not full path) of the best match, or null if cache is empty or absent.
    /// </summary>
    internal string? TrySelectFromLocalCache(string repoId, string? preferredQuantization)
    {
        var cacheDir = Path.GetDirectoryName(GetCachedPath(repoId, "placeholder.gguf"));
        if (cacheDir == null || !Directory.Exists(cacheDir))
            return null;

        var ggufFiles = Directory.EnumerateFiles(cacheDir, "*.gguf", SearchOption.TopDirectoryOnly)
            .Select(Path.GetFileName)
            .Where(f => f != null && !IsCompanionFile(f))
            .ToList();

        if (ggufFiles.Count == 0)
            return null;

        if (!string.IsNullOrEmpty(preferredQuantization))
        {
            var preferred = ggufFiles.FirstOrDefault(f =>
                f!.Contains(preferredQuantization, StringComparison.OrdinalIgnoreCase));
            if (preferred != null)
                return preferred;
        }

        foreach (var quant in DefaultQuantizationPriority)
        {
            var match = ggufFiles.FirstOrDefault(f =>
                f!.Contains(quant, StringComparison.OrdinalIgnoreCase));
            if (match != null)
                return match;
        }

        return ggufFiles[0];
    }

    /// <summary>
    /// Lists all files in a HuggingFace repository using the Tree API (includes file sizes).
    /// </summary>
    private Task<IReadOnlyList<RepoFile>> ListRepositoryFilesAsync(
        string repoId,
        CancellationToken cancellationToken)
    {
        return _discoveryService.ListRepositoryFilesAsync(repoId, "main", cancellationToken);
    }

    private ModelNotFoundException NotCached(string repoId, string file) =>
        new($"'{file}' of model '{repoId}' is not in the local cache ({Path.GetDirectoryName(GetCachedPath(repoId, "model.gguf"))}) and downloads are disabled.", repoId);

    /// <summary>
    /// Downloads a single file with resume support.
    /// </summary>
    private async Task DownloadFileAsync(
        string repoId,
        string filename,
        string destinationPath,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        // Ensure directory exists
        var dir = Path.GetDirectoryName(destinationPath);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir);
        }

        // Build download URL
        var url = $"{HuggingFaceFileBase}/{repoId}/resolve/main/{filename}";

        var tempPath = destinationPath + ".part";
        long startPosition = 0;

        // Check for partial download
        if (File.Exists(tempPath))
        {
            startPosition = new FileInfo(tempPath).Length;
        }

        // Create request with optional range header
        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        if (startPosition > 0)
        {
            request.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(startPosition, null);
        }

        using var response = await _httpClient.SendAsync(
            request,
            HttpCompletionOption.ResponseHeadersRead,
            cancellationToken);

        // Handle 416 (Range Not Satisfiable) - file already complete
        if (response.StatusCode == System.Net.HttpStatusCode.RequestedRangeNotSatisfiable)
        {
            if (File.Exists(tempPath))
            {
                // Retry: the destination may be transiently held open by another process/AV
                // scanner right after a previous attempt renamed it into place.
                await FileIoRetry.ExecuteAsync(
                    () => File.Move(tempPath, destinationPath, overwrite: true),
                    cancellationToken);
            }
            return;
        }

        response.EnsureSuccessStatusCode();

        // Determine total size
        long totalBytes = response.Content.Headers.ContentLength ?? 0;
        if (response.StatusCode == System.Net.HttpStatusCode.PartialContent)
        {
            var contentRange = response.Content.Headers.ContentRange;
            if (contentRange?.Length.HasValue == true)
            {
                totalBytes = contentRange.Length.Value;
            }
            else
            {
                totalBytes = startPosition + (response.Content.Headers.ContentLength ?? 0);
            }
        }
        else
        {
            startPosition = 0; // Full download
        }

        // Download with progress - use explicit block to ensure streams are closed before File.Move
        {
            var downloadStarted = DateTimeOffset.UtcNow;
            Trace.TraceInformation($"[GgufModelDownloader] Download started: {filename} ({totalBytes / (1024.0 * 1024 * 1024):F2} GB from {url})");

            await using var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken);
            var fileMode = startPosition > 0 ? FileMode.Append : FileMode.Create;

            // Retry: two callers racing to acquire the same ".part" file (a caller bypassing
            // the model pool's lock, or a genuinely concurrent second process) hit this as
            // IOException.
            await using var fileStream = await FileIoRetry.ExecuteAsync(
                () => new FileStream(tempPath, fileMode, FileAccess.Write, FileShare.None, 81920, true),
                cancellationToken);

            var buffer = new byte[81920];
            long bytesDownloaded = startPosition;
            int bytesRead;
            var lastLoggedAt = DateTimeOffset.UtcNow;

            while ((bytesRead = await contentStream.ReadAsync(buffer, cancellationToken)) > 0)
            {
                await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), cancellationToken);
                bytesDownloaded += bytesRead;

                progress?.Report(new DownloadProgress
                {
                    FileName = filename,
                    BytesDownloaded = bytesDownloaded,
                    TotalBytes = totalBytes
                });

                var now = DateTimeOffset.UtcNow;
                if (totalBytes > 0 && (now - lastLoggedAt).TotalSeconds >= 30)
                {
                    var pct = bytesDownloaded * 100.0 / totalBytes;
                    Trace.TraceInformation($"[GgufModelDownloader] Downloading {filename}: {pct:F1}% ({bytesDownloaded / (1024.0 * 1024):F0} MB / {totalBytes / (1024.0 * 1024):F0} MB)");
                    lastLoggedAt = now;
                }
            }

            var elapsed = DateTimeOffset.UtcNow - downloadStarted;
            Trace.TraceInformation($"[GgufModelDownloader] Download complete: {filename} ({elapsed.TotalSeconds:F0}s)");

            // Ensure data is flushed to disk
            await fileStream.FlushAsync(cancellationToken);
        }

        // Move to final location (streams are now closed). Retry: the destination path may be
        // transiently held open by another process/AV scanner immediately after this rename.
        await FileIoRetry.ExecuteAsync(
            () => File.Move(tempPath, destinationPath, overwrite: true),
            cancellationToken);
    }

    private string GetCachedPath(string repoId, string filename)
    {
        // Store in HuggingFace-compatible structure
        var safeRepoId = repoId.Replace('/', Path.DirectorySeparatorChar);
        var modelDir = "models--" + safeRepoId.Replace(Path.DirectorySeparatorChar.ToString(), "--");
        return Path.Combine(_cacheDirectory, modelDir, "snapshots", "main", filename);
    }

    private static string? ExtractQuantization(string filename)
    {
        // Common GGUF quantization patterns
        var quantPatterns = new[]
        {
            "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
            "Q4_K_S", "Q4_K_M", "Q4_0", "Q4_1",
            "Q5_K_S", "Q5_K_M", "Q5_0", "Q5_1",
            "Q6_K", "Q8_0", "F16", "F32",
            "IQ4_XS", "IQ4_NL", "IQ3_XXS", "IQ3_XS"
        };

        foreach (var pattern in quantPatterns)
        {
            if (filename.Contains(pattern, StringComparison.OrdinalIgnoreCase))
                return pattern;
        }

        return null;
    }

    private static int GetQuantizationPriority(string? quantization)
    {
        if (string.IsNullOrEmpty(quantization))
            return -1;

        var index = Array.FindIndex(DefaultQuantizationPriority,
            q => q.Equals(quantization, StringComparison.OrdinalIgnoreCase));

        return index >= 0 ? DefaultQuantizationPriority.Length - index : -1;
    }

    internal static bool IsMmprojFile(string filename) =>
        filename.StartsWith("mmproj", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Multi-token-prediction (MTP) speculative-decoding draft/assistant companion file
    /// (llama.cpp "gemma4_assistant" architecture and similar). Not a standalone chat model —
    /// must never be selected as the main model (ggml-org/llama.cpp#24343: loading one directly
    /// crashes with "requires ctx_other to be set" during llama.cpp's memory fitting).
    /// </summary>
    internal static bool IsMtpFile(string filename) =>
        filename.StartsWith("mtp", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// DFlash block-diffusion speculative-decoding draft companion file (ggml-org/llama.cpp#22105).
    /// Same class as <see cref="IsMtpFile"/> — a drafter for the target model, not usable standalone.
    /// </summary>
    internal static bool IsDflashFile(string filename) =>
        filename.StartsWith("dflash", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// True for any file that is a companion to a standalone model — a multimodal projector or a
    /// speculative-decoding draft/assistant — and must therefore never be selected as the main
    /// model in quantization/size-based auto-selection.
    /// </summary>
    internal static bool IsCompanionFile(string filename) =>
        IsMmprojFile(filename) || IsMtpFile(filename) || IsDflashFile(filename);

    public void Dispose()
    {
        if (!_disposed)
        {
            _httpClient.Dispose();
            _discoveryService.Dispose();
            _disposed = true;
        }
    }
}

/// <summary>
/// Information about a GGUF file in a repository.
/// </summary>
public sealed record GgufFileInfo
{
    /// <summary>File name (e.g., "Llama-3.2-3B-Instruct-Q4_K_M.gguf").</summary>
    public required string FileName { get; init; }

    /// <summary>Full path within repository.</summary>
    public required string Path { get; init; }

    /// <summary>File size in bytes.</summary>
    public long SizeBytes { get; init; }

    /// <summary>Detected quantization type (e.g., "Q4_K_M").</summary>
    public string? Quantization { get; init; }

    /// <summary>File size in gigabytes.</summary>
    public double SizeGB => SizeBytes / (1024.0 * 1024.0 * 1024.0);
}
