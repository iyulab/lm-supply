namespace LMSupply.Core.Download;

/// <summary>
/// Represents available hardware memory for GGUF model selection.
/// </summary>
/// <param name="VramBytes">GPU VRAM in bytes (0 for CPU-only systems).</param>
/// <param name="RamBytes">System RAM in bytes.</param>
/// <param name="ContextLength">Expected inference context length for KV cache estimation (default 4096).</param>
public record AvailableMemory(long VramBytes, long RamBytes, int ContextLength = 4096)
{
    // Reserve 2GB for GPU driver and system overhead
    private const long GpuOverheadBytes = 2L * 1024 * 1024 * 1024;
    // Reserve 4GB for OS and system processes
    private const long CpuOverheadBytes = 4L * 1024 * 1024 * 1024;
    // Add 10% runtime overhead for model loading structures
    private const double RuntimeOverheadFactor = 1.10;

    /// <summary>Usable GPU VRAM after deducting system overhead.</summary>
    public long UsableVramBytes => Math.Max(0, VramBytes - GpuOverheadBytes);

    /// <summary>Usable system RAM after deducting OS overhead.</summary>
    public long UsableRamBytes => Math.Max(0, RamBytes - CpuOverheadBytes);

    /// <summary>Total usable memory (VRAM + RAM) for hybrid GPU/CPU inference.</summary>
    public long TotalUsableBytes => UsableVramBytes + UsableRamBytes;

    /// <summary>
    /// Estimates KV cache memory from model file size and context length.
    /// Uses rough heuristics when exact model parameters are unknown.
    /// Formula: contextLength × estimatedHiddenSize × 2 (K+V) × estimatedLayers × 2 (FP16 bytes)
    /// </summary>
    public static long EstimateKvCacheBytes(long fileSizeBytes, int contextLength)
    {
        var estimatedHiddenSize = fileSizeBytes switch
        {
            < 2L * 1024 * 1024 * 1024 => 2048,
            < 5L * 1024 * 1024 * 1024 => 3072,
            < 10L * 1024 * 1024 * 1024 => 4096,
            _ => 5120
        };
        var estimatedLayers = fileSizeBytes switch
        {
            < 2L * 1024 * 1024 * 1024 => 22,
            < 5L * 1024 * 1024 * 1024 => 28,
            < 10L * 1024 * 1024 * 1024 => 32,
            _ => 40
        };
        return (long)contextLength * estimatedHiddenSize * 2 * estimatedLayers * 2;
    }

    /// <summary>
    /// Returns true if the model and a KV cache of <paramref name="kvCacheBytes"/> (known for the model, rather than
    /// estimated from the file size) fit entirely in GPU VRAM.
    /// </summary>
    internal bool FitsInGpu(long fileSizeBytes, long kvCacheBytes)
        => VramBytes > 0 && (long)(fileSizeBytes * RuntimeOverheadFactor) + kvCacheBytes <= UsableVramBytes;

    /// <summary>
    /// Returns true if the model and a KV cache of <paramref name="kvCacheBytes"/> fit in total available memory.
    /// </summary>
    internal bool FitsInMemory(long fileSizeBytes, long kvCacheBytes)
        => (long)(fileSizeBytes * RuntimeOverheadFactor) + kvCacheBytes <= TotalUsableBytes;

    /// <summary>
    /// Returns true if the model + KV cache fits entirely in GPU VRAM.
    /// </summary>
    public bool FitsInGpu(long fileSizeBytes)
    {
        var total = (long)(fileSizeBytes * RuntimeOverheadFactor) + EstimateKvCacheBytes(fileSizeBytes, ContextLength);
        return VramBytes > 0 && total <= UsableVramBytes;
    }

    /// <summary>
    /// Returns true if the model + KV cache fits in total available memory (GPU + RAM).
    /// </summary>
    public bool FitsInMemory(long fileSizeBytes)
    {
        var total = (long)(fileSizeBytes * RuntimeOverheadFactor) + EstimateKvCacheBytes(fileSizeBytes, ContextLength);
        return total <= TotalUsableBytes;
    }
}

/// <summary>
/// Selects the optimal GGUF file group based on hardware memory constraints.
///
/// Algorithm:
/// 1. Filter groups that fit in available memory (size × 1.1 ≤ total usable memory)
/// 2. If preferred quantization specified and fits, select it
/// 3. Otherwise, select the largest fitting group (larger file = higher quality for same architecture)
/// 4. If nothing fits, throw a descriptive exception
/// </summary>
public static class GgufFileSelector
{
    /// <summary>
    /// Selects the best GGUF file group for the given hardware memory constraints.
    /// </summary>
    /// <param name="groups">Available file groups (e.g. from GgufFileGroup.GroupFiles).</param>
    /// <param name="memory">Available hardware memory specification.</param>
    /// <param name="preferredQuantization">
    /// Optional user-specified quantization type (e.g. "Q8_0", "Q6_K").
    /// If the preferred type fits in memory, it is selected. Otherwise falls back to auto.
    /// </param>
    /// <param name="vramOnly">
    /// When <c>true</c> and the host has VRAM, candidates are filtered by VRAM-only fit
    /// (model + KV cache ≤ usable VRAM) instead of total VRAM+RAM. Use this on GPU hosts
    /// to avoid selecting variants (e.g. bf16) that only fit because system RAM is large.
    /// If nothing fits the VRAM budget, the smallest variant is returned as a fallback
    /// (matches GgufModelRegistry's FallbackToSmallest behavior). Default <c>false</c>
    /// preserves the legacy VRAM+RAM total-memory fit logic.
    /// </param>
    /// <returns>The best fitting file group.</returns>
    /// <exception cref="ArgumentException">Thrown when groups is empty.</exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when no file fits in available memory, with details about required vs available memory.
    /// </exception>
    public static GgufFileGroup Select(
        IEnumerable<GgufFileGroup> groups,
        AvailableMemory memory,
        string? preferredQuantization = null,
        bool vramOnly = false)
    {
        var list = groups.ToList();

        if (list.Count == 0)
            throw new ArgumentException("No GGUF file groups provided.", nameof(groups));

        // VRAM-only mode prevents bf16 (or any oversized variant) being chosen on a
        // small-VRAM host because total RAM happens to be large. Without this, a
        // 4GB-VRAM laptop with 32GB RAM would happily pick a 15GB bf16 variant since
        // it "fits in memory", silently OOMing at load.
        var fitsInBudget = vramOnly && memory.VramBytes > 0
            ? (Func<long, bool>)memory.FitsInGpu
            : memory.FitsInMemory;

        var fitting = list
            .Where(g => fitsInBudget(g.TotalSizeBytes))
            .OrderByDescending(g => g.TotalSizeBytes)
            .ToList();

        if (fitting.Count == 0)
        {
            // VRAM-only mode: fall back to smallest variant (partial CPU offload via
            // llama-server is acceptable) rather than throwing. Mirrors the registry's
            // GetAutoSelection FallbackToSmallest behavior and prevents the v0.29.0
            // failure mode where bf16 was silently chosen via the RAM budget.
            if (vramOnly)
                return list.OrderBy(g => g.TotalSizeBytes).First();

            var smallest = list.MinBy(g => g.TotalSizeBytes)!;
            var availableGB = memory.TotalUsableBytes / (1024.0 * 1024 * 1024);
            throw new InvalidOperationException(
                $"No GGUF file fits in available memory. " +
                $"Smallest option is {smallest.TotalSizeGB:F1}GB ({smallest.PrimaryFileName}), " +
                $"but only {availableGB:F1}GB is available. " +
                "Consider using a smaller model or freeing system memory.");
        }

        // If user specified a preferred quantization and it fits, use it
        if (!string.IsNullOrWhiteSpace(preferredQuantization))
        {
            var preferred = fitting.FirstOrDefault(g =>
                MatchesQuantization(g.PrimaryFileName, preferredQuantization));

            if (preferred is not null)
                return preferred;
            // Falls through to auto selection if preferred doesn't fit or not found
        }

        // Auto: return the largest fitting group (= highest quality within memory budget)
        return fitting[0];
    }

    /// <summary>
    /// Checks if a filename contains the given quantization type.
    /// Uses case-insensitive substring matching to handle various naming conventions:
    /// - Standard: model-Q4_K_M.gguf
    /// - iMatrix: model-Q4_K_M-imat.gguf
    /// - Dot separator: Meta-Llama-3-8B.Q4_K_M.gguf
    /// - Underscore separator: model_Q4_K_M.gguf
    /// </summary>
    public static bool MatchesQuantization(string filename, string quantization) =>
        filename.Contains(quantization, StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Creates an AvailableMemory instance from a HardwareProfile.
    /// </summary>
    public static AvailableMemory FromHardwareProfile(Hardware.HardwareProfile profile, int contextLength = 4096) =>
        new(
            VramBytes: profile.GpuInfo.EffectiveAvailableBytes ?? 0,
            RamBytes: profile.SystemMemoryBytes,
            ContextLength: contextLength);
}
