using LMSupply.Generator.Internal.Llama;
using LMSupply.Generator.Models;

namespace LMSupply.Generator.Abstractions;

/// <summary>
/// Represents a loaded text generation model.
/// </summary>
public interface IGeneratorModel : ITextGenerator
{
    /// <summary>
    /// Gets the maximum context length supported by the model.
    /// </summary>
    int MaxContextLength { get; }

    /// <summary>
    /// Gets the chat formatter for this model.
    /// </summary>
    IChatFormatter ChatFormatter { get; }

    /// <summary>
    /// Gets whether GPU acceleration is being used for inference.
    /// </summary>
    bool IsGpuActive { get; }

    /// <summary>
    /// Gets the list of active execution providers.
    /// </summary>
    IReadOnlyList<string> ActiveProviders { get; }

    /// <summary>
    /// Gets the execution provider that was requested.
    /// </summary>
    ExecutionProvider RequestedProvider { get; }

    /// <summary>
    /// Gets the estimated memory usage of this model in bytes.
    /// Uses MemoryEstimator to calculate based on model parameters, context length, and quantization.
    /// </summary>
    long? EstimatedMemoryBytes { get; }

    /// <summary>
    /// Gets information about the model.
    /// </summary>
    GeneratorModelInfo GetModelInfo();

    /// <summary>
    /// Counts the exact number of tokens in the given text using the model's tokenizer.
    /// </summary>
    /// <param name="text">The text to tokenize and count.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The exact token count.</returns>
    Task<int> CountTokensAsync(string text, CancellationToken cancellationToken = default);

    /// <summary>
    /// Counts the exact number of tokens for formatted chat messages using the model's tokenizer.
    /// Messages are formatted using the model's chat template before tokenization.
    /// </summary>
    /// <param name="messages">The chat messages to format and count.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The exact token count of the formatted prompt.</returns>
    Task<int> CountTokensAsync(IEnumerable<ChatMessage> messages, CancellationToken cancellationToken = default);
}

/// <summary>
/// Information about a generator model.
/// </summary>
/// <param name="ModelId">The loaded model: a registry alias's display name, naming the loaded quantization when
/// another than the alias's default was loaded (see <see cref="IsQuantizationSubstituted"/>); otherwise the requested id.</param>
/// <param name="ModelPath">The local path to the model files.</param>
/// <param name="MaxContextLength">Maximum context length.</param>
/// <param name="ChatFormat">The chat format name.</param>
/// <param name="ExecutionProvider">The execution provider being used.</param>
public readonly record struct GeneratorModelInfo(
    string ModelId,
    string ModelPath,
    int MaxContextLength,
    string ChatFormat,
    string ExecutionProvider) : IModelInfoBase
{
    /// <summary>
    /// Gets GGUF-specific metadata (only available for GGUF models).
    /// </summary>
    public GgufMetadata? GgufMetadata { get; init; }

    /// <summary>
    /// Gets the backend startup log for diagnostics (llama-server only).
    /// </summary>
    public string? BackendLog { get; init; }

    /// <summary>
    /// Gets the runtime version (e.g., llama-server "b7898").
    /// </summary>
    public string? RuntimeVersion { get; init; }

    /// <summary>
    /// Gets auto-selection diagnostics (VRAM total/free/budget, safety margin, reason).
    /// Null when the model was loaded via an explicit alias rather than the auto path.
    /// </summary>
    public SelectionDiagnostics? Diagnostics { get; init; }

    /// <summary>
    /// Gets the VRAM-capped context length actually sent to llama-server.
    /// Non-null only when VRAM constraints reduced the context below <see cref="MaxContextLength"/>.
    /// Null when the requested context was fully honoured or no adjustment occurred.
    /// </summary>
    public int? AdjustedContextLength { get; init; }

    /// <summary>
    /// Number of transformer layers offloaded to GPU. Null when all layers fit in VRAM or the model
    /// was loaded on CPU. Non-null only when partial offload occurred (GpuLayers &lt; TotalLayers).
    /// </summary>
    public int? GpuLayers { get; init; }

    /// <summary>
    /// Total transformer layer count used for offload ratio calculation. Non-null when GpuLayers is set.
    /// </summary>
    public int? TotalLayers { get; init; }

    /// <summary>
    /// Estimated VRAM usage in bytes for the partial-offload configuration. Non-null when GpuLayers is set.
    /// </summary>
    public long? EstimatedVramBytes { get; init; }

    /// <summary>
    /// Estimated RAM usage in bytes for layers not offloaded to GPU. Non-null when GpuLayers is set.
    /// </summary>
    public long? EstimatedRamBytes { get; init; }

    /// <summary>
    /// VRAM budget used for the context estimate — <c>VramBudget.GetAvailableBytes</c> result
    /// (after the <c>LMSUPPLY_VRAM_BUDGET_MB</c> override + safety margin), in bytes.
    /// Null on the CPU path. Distinguishes an accurately-small VRAM budget from an under-reported one.
    /// </summary>
    public long? VramBudgetBytes { get; init; }

    /// <summary>
    /// GPU-reported free VRAM at load time, in bytes. Null on the CPU path.
    /// </summary>
    public long? VramFreeBytes { get; init; }

    /// <summary>
    /// GPU-reported total VRAM at load time, in bytes. Null on the CPU path.
    /// </summary>
    public long? VramTotalBytes { get; init; }

    /// <summary>
    /// True when the VRAM-derived safe-context estimate fell below the 512-token usable floor
    /// (VRAM insufficient for a usable context) — a discrete signal, distinct from a legitimately
    /// small 512-token request. Consumers should prefer this over inferring brick state from a
    /// magic-number 512 in the logs. Set even when Auto subsequently fell back to CPU (the GPU
    /// estimate was floored, which is what triggered the fallback).
    /// </summary>
    public bool ContextFlooredByVram { get; init; }

    /// <summary>
    /// Gets model-specific known issues from the registry (e.g., "tool-use-unreliable-q4").
    /// Empty when no issues are registered or the model was loaded via an explicit path.
    /// </summary>
    public IReadOnlyList<string> KnownIssues { get; init; } = [];

    /// <summary>
    /// Gets the id the load was asked for: a registry alias (<c>gguf:gemma4-balanced</c>), a HuggingFace repo id,
    /// or a path. <see cref="ModelId"/> describes what was loaded; this is what was requested.
    /// Set by GGUF (llama-server) loads; <see langword="null"/> otherwise.
    /// </summary>
    public string? RequestedModelId { get; init; }

    /// <summary>
    /// Gets the file name of the GGUF the load opened (the first shard of a split model).
    /// Set by GGUF (llama-server) loads; <see langword="null"/> otherwise.
    /// </summary>
    public string? LoadedFile { get; init; }

    /// <summary>
    /// Gets the default file of the requested registry alias — what the load opens when that file fits this host.
    /// <see langword="null"/> when no registry alias was requested.
    /// </summary>
    public string? RequestedFile { get; init; }

    /// <summary>
    /// Whether the load opened another quantization than the requested alias's default — the default did not
    /// fit the memory budget and a smaller one stood in. <see cref="ModelId"/> then names the loaded
    /// quantization, and <see cref="RequestedFile"/> / <see cref="LoadedFile"/> say what was replaced by what.
    /// </summary>
    public bool IsQuantizationSubstituted =>
        RequestedFile is not null && LoadedFile is not null &&
        !string.Equals(RequestedFile, LoadedFile, StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Gets the model identifier (IModelInfoBase.Id).
    /// </summary>
    string IModelInfoBase.Id => ModelId;

    /// <summary>
    /// Gets the requested alias, or <see cref="ModelId"/> when the load did not record one.
    /// </summary>
    string IModelInfoBase.AliasName => RequestedModelId ?? ModelId;

    /// <summary>
    /// Gets the model description.
    /// </summary>
    string? IModelInfoBase.Description => $"{ChatFormat} model at {ModelPath}";

    /// <summary>
    /// Gets a summary of the model's architecture from GGUF metadata.
    /// </summary>
    public string? GetArchitectureSummary()
    {
        if (GgufMetadata == null)
            return null;

        return GgufMetadata.GetSummary();
    }

    /// <summary>
    /// Gets the estimated parameter count from GGUF metadata.
    /// </summary>
    public long? ParameterCount => GgufMetadata?.EstimatedParameterCount;

    /// <summary>
    /// Gets the quantization type from GGUF metadata.
    /// </summary>
    public string? QuantizationType => GgufMetadata?.QuantizationType;

    /// <summary>
    /// Gets the model architecture from GGUF metadata (e.g., "llama", "gemma").
    /// </summary>
    public string? Architecture => GgufMetadata?.Architecture;
}
