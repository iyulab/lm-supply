using System.Diagnostics;
using LMSupply.Core.Download;
using LMSupply.Decider.Internal;
using LMSupply.Llama.Gguf;
using LMSupply.Llama.Server;

namespace LMSupply.Decider;

/// <summary>
/// Main entry point for loading a model that answers typed decisions.
/// </summary>
/// <remarks>
/// Any GGUF model llama-server can run is a decider. A native decision model (one llama-server reports with the
/// <c>decisions</c> output modality) answers through the server's decision endpoint; an instruct model answers through
/// one-token chat completions whose answer-label probabilities are read and renormalized
/// (<see cref="IDeciderModel.Path"/> says which).
/// </remarks>
public static class LocalDecider
{
    /// <summary>
    /// The quantization a GGUF repository is loaded at when it offers several and the caller names none.
    /// </summary>
    internal const string DefaultGgufQuantization = "Q4_K_M";

    /// <summary>
    /// Loads a decider.
    /// </summary>
    /// <param name="modelIdOrPath">
    /// A local <c>.gguf</c> file, or a HuggingFace repository holding GGUF files (<c>org/repo</c>, optionally prefixed
    /// <c>gguf:</c>). From a repository the file at <see cref="LMSupplyOptionsBase.QuantizationHint"/> is loaded
    /// (<c>Q4_K_M</c> when not set), or the largest that fits when there is none.
    /// </param>
    /// <param name="options">Optional configuration.</param>
    /// <param name="progress">Optional progress reporting for downloads and the server start.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded decider.</returns>
    /// <exception cref="ModelNotFoundException">The model is neither a local file nor a repository, or (downloads disabled) is not cached.</exception>
    public static async Task<IDeciderModel> LoadAsync(
        string modelIdOrPath,
        DeciderOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        progress = LoadProgress.Begin(progress);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelIdOrPath);
        options = options?.Clone() ?? new DeciderOptions();
        options.Validate();
        ExecutionProviderSupport.ThrowIfUnsupported(options.Provider);

        var (modelId, modelPath) = await ResolveModelFileAsync(modelIdOrPath, options, progress, cancellationToken)
            .ConfigureAwait(false);

        GgufMetadata? metadata = null;
        try
        {
            metadata = await GgufMetadataReader.ReadAsync(modelPath, false, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is not OperationCanceledException || !cancellationToken.IsCancellationRequested)
        {
            Trace.TraceInformation($"[LocalDecider] GGUF metadata reading failed: {ex.Message}");
        }

        var lease = await LeaseServerAsync(modelPath, metadata, options, progress, cancellationToken).ConfigureAwait(false);
        try
        {
            return await LlamaServerDeciderModel.CreateAsync(
                modelId,
                async ct => (await lease.GetAsync(ct).ConfigureAwait(false)).Client,
                lease,
                options.Temperature,
                cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            await lease.DisposeAsync().ConfigureAwait(false);
            throw;
        }
    }

    /// <summary>
    /// The minimum-build key for a model with <paramref name="decisionType"/> (from its GGUF metadata): native decision
    /// models need the build that serves <c>/v1/systemone</c>, and the clef type a later one. Null for other models.
    /// </summary>
    internal static string? VersionRequirementKey(string? decisionType) => decisionType switch
    {
        null or "" => null,
        _ when string.Equals(decisionType, "clef", StringComparison.OrdinalIgnoreCase) => "decisions-clef",
        _ => "decisions",
    };

    /// <summary>The context size a load asks for: the caller's, else the default, capped at the model's own.</summary>
    internal static int ResolveContextSize(int? requested, int? modelContext)
    {
        if (requested is { } r)
            return r;
        return modelContext is > 0 and < DeciderOptions.DefaultContextSize ? modelContext.Value : DeciderOptions.DefaultContextSize;
    }

    private static async Task<RestartingLease<ServerLease>> LeaseServerAsync(
        string modelPath,
        GgufMetadata? metadata,
        DeciderOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        progress?.Report(new DownloadProgress
        {
            FileName = "llama-server",
            Kind = DownloadKind.Runtime,
            Phase = DownloadPhase.Preparing
        });

        var preferredBackend = global::LMSupply.Llama.LlamaBackendSelector.MapProvider(
            options.Provider, Hardware.HardwareProfile.For(options.Provider).GpuInfo);
        var updateService = LlamaServerUpdateService.Resolve(options.ServerUpdateOptions);
        // With downloads disabled the server comes from the cache only, with no update check: the load makes no request.
        var installation = await updateService.AcquireServerAsync(
            preferredBackend,
            options.DisableAutoDownload,
            progress,
            cancellationToken).ConfigureAwait(false);

        var serverPath = installation.ServerPath;
        var backend = installation.Backend;
        var serverVersion = installation.Version;

        // A native decision model needs a build that serves it. A cached build that is too old is updated once
        // (a pinned version is never exceeded, so Validate then reports it); with downloads disabled it is reported.
        var native = metadata?.DecisionType is not null;
        if (VersionRequirementKey(metadata?.DecisionType) is { } requirement)
        {
            if (!options.DisableAutoDownload && !LlamaServerVersionRequirements.MeetsMinimum(serverVersion, requirement))
            {
                var retry = await updateService.CheckAndApplyUpdateAsync(preferredBackend, progress, cancellationToken)
                    .ConfigureAwait(false);
                serverPath = retry.ServerPath;
                backend = retry.Backend;
                serverVersion = retry.NewVersion ?? retry.PreviousVersion;
            }

            LlamaServerVersionRequirements.Validate(serverVersion, requirement);
        }

        var contextSize = ResolveContextSize(options.ContextSize, metadata?.ContextLength);
        var serverConfig = new LlamaServerConfig
        {
            ModelPath = modelPath,
            Port = 0, // Auto-assign
            ContextSize = contextSize,
            GpuLayers = backend == LlamaServerBackend.Cpu ? 0 : -1,
            // Native decision models evaluate a whole question prompt in one physical batch, so the batch is the
            // context; generation keeps the default batches.
            BatchSize = native ? contextSize : 512,
            UBatchSize = native ? contextSize : null,
            // One slot: the emulated path sends a call's questions one after another, and they share the state's
            // prompt prefix only within a slot.
            Parallel = 1,
            Mode = ServerMode.Generation,
            ServerVersion = serverVersion,
            ShutdownTimeout = TimeSpan.FromSeconds(10)
        };

        var serverLease = await LlamaServerPool.Instance.LeaseAsync(
            serverPath,
            serverConfig,
            backend,
            progress,
            cancellationToken).ConfigureAwait(false);

        return RestartingServerLease.Create(
            serverLease,
            ct => LlamaServerPool.Instance.LeaseAsync(serverPath, serverConfig, backend, null, ct),
            $"LocalDecider '{Path.GetFileName(modelPath)}'");
    }

    private static async Task<(string ModelId, string ModelPath)> ResolveModelFileAsync(
        string modelIdOrPath,
        DeciderOptions options,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var target = modelIdOrPath.StartsWith("gguf:", StringComparison.OrdinalIgnoreCase) ? modelIdOrPath[5..] : modelIdOrPath;

        if (File.Exists(target))
        {
            progress?.Report(DownloadProgress.FileInPlace(Path.GetFileName(target), target));
            return (Path.GetFileNameWithoutExtension(target), target);
        }

        var repoId = target;
        if (!repoId.Contains('/') || repoId.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
        {
            throw new ModelNotFoundException(
                $"Decider model not found: '{modelIdOrPath}'. Provide a local path to a .gguf file or a HuggingFace " +
                "repository holding GGUF files (e.g. 'org/model-GGUF').",
                modelIdOrPath);
        }

        var quantization = string.IsNullOrWhiteSpace(options.QuantizationHint) ? DefaultGgufQuantization : options.QuantizationHint;
        var cacheDir = options.CacheDirectory ?? CacheManager.GetDefaultCacheDirectory();
        using var downloader = new GgufDownloader(cacheDir, localFilesOnly: options.DisableAutoDownload);
        var path = await downloader.DownloadAsync(repoId, quantization, progress, cancellationToken).ConfigureAwait(false);
        return (repoId.Split('/').Last(), path);
    }
}
