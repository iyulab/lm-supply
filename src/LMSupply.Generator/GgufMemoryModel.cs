namespace LMSupply.Generator;

/// <summary>
/// The one rule for whether a GGUF model fits a GPU's memory: automatic selection, the GPU layer fit and the
/// context bound of a load all judge with it, so a model the selector recommends is the model the loader runs
/// on the GPU the selector judged.
/// </summary>
/// <remarks>
/// <para>
/// What a load holds on the GPU is the weights (the file, plus a tenth for the loading structures and the
/// mapped part), the KV cache, and a fixed headroom for the runtime and the compute scratch. The usable memory it
/// is held against already leaves the safety margin (<see cref="LMSupply.Hardware.VramBudget"/>), so the margin
/// is counted once, there.
/// </para>
/// <para>
/// A mixture-of-experts model reserves one more fixed amount for its expert-activation scratch and recurrent
/// state. It is a fixed amount, not a share of the budget: the scratch follows the model and the batch, not how
/// much memory the machine has. Measured on Metal with Qwen3.6-35B-A3B (IQ4_XS, llama.cpp b10331, full
/// offload, default micro-batch): model buffers 16,899 MiB for a 16,909 MiB file, KV 640 MiB at 32,768 tokens
/// (80 MiB at 4,096), recurrent state 251 MiB, compute buffer 106 MiB (78 MiB at 4,096) — about 350 MiB beyond
/// the weights and the KV cache. The reserve is about three times that. CUDA has not been measured; a load that
/// still runs out of memory is retried with fewer GPU layers.
/// </para>
/// </remarks>
internal static class GgufMemoryModel
{
    /// <summary>The weights in memory against the file: loading structures and the mapped part.</summary>
    internal const double WeightsFactor = 1.1;

    /// <summary>Fixed headroom for the runtime and the compute scratch.</summary>
    internal const long HeadroomBytes = 512L * 1024 * 1024;

    /// <summary>Extra fixed reserve of a mixture-of-experts model (expert-activation scratch, recurrent state).</summary>
    internal const long MixtureOfExpertsReserveBytes = 1024L * 1024 * 1024;

    /// <summary>
    /// What the model holds on the GPU apart from its KV cache: <paramref name="gpuShare"/> of the weights (1 = every
    /// layer on the GPU), the headroom, and the reserve of a mixture-of-experts model.
    /// </summary>
    internal static long ResidentBytes(long modelFileBytes, bool mixtureOfExperts, double gpuShare = 1.0)
        => (long)(modelFileBytes * WeightsFactor * Math.Clamp(gpuShare, 0.0, 1.0))
           + HeadroomBytes
           + (mixtureOfExperts ? MixtureOfExpertsReserveBytes : 0);

    /// <summary>The memory left for the KV cache once the model is resident; 0 when the model alone does not fit.</summary>
    internal static long KvBudgetBytes(long usableGpuBytes, long modelFileBytes, bool mixtureOfExperts, double gpuShare = 1.0)
        => Math.Max(0, usableGpuBytes - ResidentBytes(modelFileBytes, mixtureOfExperts, gpuShare));

    /// <summary>Whether the model and a KV cache of <paramref name="kvCacheBytes"/> fit in <paramref name="usableGpuBytes"/>.</summary>
    internal static bool Fits(long usableGpuBytes, long modelFileBytes, long kvCacheBytes, bool mixtureOfExperts, double gpuShare = 1.0)
        => usableGpuBytes > 0
           && usableGpuBytes - ResidentBytes(modelFileBytes, mixtureOfExperts, gpuShare) >= kvCacheBytes;

    /// <summary>
    /// The share of the weights a load puts on the GPU: all of them when every layer is offloaded (<c>-1</c>) or the
    /// layer count is not known, else the offloaded layers over the model's layers.
    /// </summary>
    internal static double GpuShare(int gpuLayerCount, int? totalLayers)
        => gpuLayerCount < 0 || totalLayers is not > 0
            ? 1.0
            : Math.Clamp(gpuLayerCount / (double)totalLayers.Value, 0.0, 1.0);
}
