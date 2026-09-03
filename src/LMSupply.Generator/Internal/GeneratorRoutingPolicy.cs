using LMSupply.Runtime;

namespace LMSupply.Generator.Internal;

/// <summary>
/// Single source of truth for the auto path's ONNX (DirectML) vs GGUF (llama.cpp) routing decision,
/// shared by <see cref="LocalGenerator"/> and <see cref="ModelFormatDetector"/>.
/// </summary>
internal static class GeneratorRoutingPolicy
{
    /// <summary>
    /// Returns true when the auto path should use the ONNX/DirectML backend instead of GGUF.
    /// ONNX/DirectML is preferred only for a <b>discrete</b> non-NVIDIA GPU on Windows (Arc, Radeon).
    /// Integrated GPUs are excluded: their reported dedicated VRAM is unreliable (shared system
    /// memory), so DirectML on them is not a real acceleration win — they route to GGUF, which
    /// resolves a CPU backend, RAM-aware model selection, and quantization downscale (see
    /// <c>LlamaBackendSelector</c> / <c>GgufModelDownloader</c>). NVIDIA and Apple use GGUF
    /// (CUDA / Metal); CPU-only uses GGUF CPU.
    /// </summary>
    /// <remarks>
    /// Also requires <see cref="OnnxGeneratorBackendRegistry.IsRegistered"/>: a consumer that only
    /// referenced LMSupply.Generator (GGUF/llama-server only, no LMSupply.Generator.Onnx package)
    /// must never have the zero-config "auto" path silently pick a backend it cannot load — it
    /// falls through to GGUF instead, same as any other hardware profile GGUF already covers.
    /// </remarks>
    public static bool ShouldUseOnnx(GpuInfo gpu, ExecutionProvider recommendedProvider)
        => OnnxGeneratorBackendRegistry.IsRegistered
           && recommendedProvider == ExecutionProvider.DirectML
           && gpu.Vendor != GpuVendor.Nvidia
           && !gpu.IsIntegrated;
}
