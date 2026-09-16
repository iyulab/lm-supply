# GPU Providers Guide

This guide explains execution providers in LMSupply and how to optimize GPU acceleration.

## Overview

LMSupply uses ONNX Runtime for inference, which supports multiple execution providers for hardware acceleration.

> **DirectML removed (0.67.0).** ONNX Runtime 1.25+ ships no DirectML execution provider and the
> `Microsoft.ML.OnnxRuntime.DirectML` package line ends at 1.24.4, so no build of LMSupply on the
> current runtime (1.30.0) can provision it. `ExecutionProvider.DirectML` is obsolete: an explicit
> request throws `NotSupportedException` on every path (ONNX session, GenAI, llama-server), and `Auto`
> no longer tries it — on a Windows machine without CUDA, ONNX sessions run on CPU and the library
> says so once per process in `Trace`. GGUF/llama-server paths still use the GPU through Vulkan
> (`LlamaBackendSelector`). A machine that had cached the 1.24.4 native from an older release was
> running a managed 1.30.0 runtime against a 1.24.4 provider binary, which is not a supported
> combination; it moves to CPU for ONNX sessions on upgrade.

---

## 1. Available Providers

| Provider | Platform | GPU Vendor | Notes |
|----------|----------|------------|-------|
| **CUDA** | Windows/Linux | NVIDIA | Best performance for NVIDIA GPUs |
| **CoreML** | macOS | Apple Silicon | Native Apple acceleration |
| **CPU** | All | N/A | Fallback, always available |

---

## 2. Provider Selection

### 2.1 Auto Detection (Default)

By default, LMSupply automatically selects the best provider:

```csharp
// Auto-detection (recommended)
await using var model = await LocalEmbedder.LoadAsync("default");
```

**Auto-detection priority:**
1. **CUDA** - If NVIDIA GPU with 4GB+ VRAM detected
2. **CoreML** - If macOS with Apple Silicon
3. **CPU** - Fallback

### 2.2 Explicit Provider Selection

Force a specific provider:

```csharp
var options = new EmbedderOptions
{
    Provider = ExecutionProvider.Cuda
};

await using var model = await LocalEmbedder.LoadAsync("default", options);
```

Available values:
- `ExecutionProvider.Auto` (default)
- `ExecutionProvider.Cuda`
- `ExecutionProvider.CoreML`
- `ExecutionProvider.Cpu`

---

## 3. Provider Details

### 3.1 CUDA (NVIDIA)

**Requirements:**
- NVIDIA GPU with Compute Capability 3.5+
- NVIDIA Driver 450.80.02+ (Linux) / 452.39+ (Windows)
- Minimum 4GB VRAM recommended

**Performance Characteristics:**
- Fastest inference for most models
- Excellent batch processing
- Low latency

**Troubleshooting:**
```csharp
// Check if CUDA is available
var profile = HardwareProfile.Current;
if (profile.GpuInfo.Vendor == GpuVendor.Nvidia)
{
    Console.WriteLine($"NVIDIA GPU: {profile.GpuInfo.DeviceName}");
    Console.WriteLine($"VRAM: {profile.GpuMemoryGB:F1} GB");
}
```

### 3.2 AMD / Intel GPUs on Windows

There is no ONNX execution provider for these GPUs on ONNX Runtime 1.25+ (DirectML was it — see the
note at the top). ONNX-backed modules (embedder, reranker, transcriber, OCR, …) run on CPU there;
the GGUF/llama-server modules (generator, and the embedder/reranker GGUF paths) use the GPU through
**Vulkan**, selected automatically under `ExecutionProvider.Auto` (see [llama.md](llama.md)).

### 3.3 CoreML (macOS)

**Requirements:**
- macOS 11.0+
- Apple Silicon (M1/M2/M3) or Intel Mac with AMD GPU

**Performance Characteristics:**
- Optimized for Apple Silicon
- Neural Engine acceleration
- Good power efficiency

### 3.4 CPU

**When used:**
- No GPU available
- GPU initialization fails
- Explicit selection

**Optimization tips:**
- Uses all available CPU cores
- Benefits from AVX2/AVX-512 instructions
- Consider smaller models for faster inference

---

## 4. Hardware Detection

### 4.1 HardwareProfile

LMSupply provides unified hardware detection:

```csharp
var profile = HardwareProfile.Current;

Console.WriteLine($"GPU: {profile.GpuInfo.DeviceName}");
Console.WriteLine($"GPU Memory: {profile.GpuMemoryGB:F1} GB");
Console.WriteLine($"System Memory: {profile.SystemMemoryGB:F1} GB");
Console.WriteLine($"Recommended Provider: {profile.RecommendedProvider}");
Console.WriteLine($"Performance Tier: {profile.Tier}");
```

### 4.2 Performance Tiers

| Tier | Criteria | Recommended Models |
|------|----------|-------------------|
| **Low** | CPU only or GPU < 4GB | Small/fast models |
| **Medium** | GPU 4-8GB or CPU 16GB+ | Base models |
| **High** | GPU 8-16GB | Large models |
| **Ultra** | GPU 16GB+ | Largest models |

---

## 5. Provider Fallback

LMSupply implements automatic fallback:

```
Requested Provider → Available? → Use
        ↓ No
   Next Provider → Available? → Use
        ↓ No
       CPU (always available)
```

**Fallback chain:**
1. CUDA → CPU (Windows, Linux)
2. CoreML → CPU (macOS)

### 5.1 Runtime Recovery (after the session is loaded)

The chain above is walked at **load time** — it answers "which provider can create a session".
A provider can still fail **at run time**: an unsupported kernel throws from the first inference,
or a cold GPU initialization hangs inside the native call and never returns. Modules that own
an ONNX session (`RecoverableOnnxSession` in `LMSupply.Core`) handle both the same way:

| Run-time failure | What happens |
|---|---|
| The provider throws (`OnnxRuntimeException`) | The provider is blacklisted, the session is recreated on the next provider in the chain, and the run is retried **once**. |
| The run exceeds the inference bound (`InferenceTimeoutException`, 60 s by default) | Same move to the next provider, then **one** retry. The hung native call is abandoned — it is never disposed underneath, so it cannot corrupt the replacement session. |
| The next provider also fails, or CPU is already active | The original exception surfaces unchanged. Nothing below CPU to fall back to. |
| `Provider = ExecutionProvider.Cpu` was requested | Recovery is off — you asked for CPU and get exactly that. |

Each recovery writes a `Trace` warning of the form
`[<Module>] Inference failed on Cuda (...)` / `[<Module>] Inference timed out on Cuda after 60s ...`
followed by `[<Module>] Recovered: now running on CPUExecutionProvider.` Attach a
`TraceListener` (see `samples/EmbedderSample`) if you want to see them. After a recovery,
`IsGpuActive` / `ActiveProviders` on the model reflect the provider actually in use.

A model made of several sessions (an encoder + decoder pair, for example) shares one provider
blacklist across them: once one session has left a provider — at run time, or because the load-time
chain already saw that provider fail for this model — its siblings leave it before their next run
instead of hitting the same crash or hang themselves. For an autoregressive decoder the bound and the
recovery apply **per decode step**, so a long output is never cut off by a whole-loop timeout.

Every `InferenceSession`-backed module is on this recovery path: Embedder, Reranker, Segmenter (SegFormer
and MobileSAM), Ocr (detection and recognition), Synthesizer, Detector, Transcriber, Translator, Captioner
(each encoder + decoder), and ImageGenerator (text encoder + UNet + VAE, one shared blacklist — a UNet
denoising step is one recoverable run). The GenAI-backed text generator (`LMSupply.Generator.Onnx`) does
not use `InferenceSession` and is outside this mechanism.

---

## 6. Best Practices

### 6.1 Let Auto-Detection Work

```csharp
// Recommended: Trust auto-detection
var options = new EmbedderOptions
{
    Provider = ExecutionProvider.Auto  // Default
};
```

### 6.2 Check Actual Provider Used

```csharp
await using var model = await LocalTranscriber.LoadAsync("default");

// Verify which provider is active
Console.WriteLine($"GPU Active: {model.IsGpuActive}");
Console.WriteLine($"Providers: {string.Join(", ", model.ActiveProviders)}");
```

### 6.3 Handle Fallback Gracefully

```csharp
await using var model = await LocalEmbedder.LoadAsync("default");
var info = model.GetModelInfo();

if (info.RequestedProvider != ExecutionProvider.Cpu &&
    !info.ActiveProviders.Contains("CUDAExecutionProvider") &&
    !info.ActiveProviders.Contains("CoreMLExecutionProvider"))
{
    Console.WriteLine("Warning: Running on CPU fallback");
}
```

### 6.4 GPU Memory Management

```csharp
// For limited VRAM, use sequential loading
await using (var embedder = await LocalEmbedder.LoadAsync("default"))
{
    // Process with embedder
}
// VRAM freed

await using (var generator = await LocalGenerator.LoadAsync("auto"))
{
    // Process with generator
}
```

---

## 7. Comparison

| Aspect | CUDA | CoreML | CPU |
|--------|------|--------|-----|
| **Speed** | Fastest | Fast | Slowest |
| **Latency** | Lowest | Low | Highest |
| **Batch Perf** | Excellent | Good | Moderate |
| **Memory** | GPU VRAM | Unified | System RAM |
| **Setup** | Driver only | Auto | None |

---

## 8. Summary

- **Auto-detection** handles most cases correctly
- **CUDA** is best for NVIDIA GPUs
- **AMD / Intel GPUs on Windows** accelerate the GGUF/llama-server paths (Vulkan); ONNX sessions run on CPU
- **CoreML** is optimal for Apple Silicon
- **CPU** is always available as fallback
- Use `HardwareProfile.Current` to check detected hardware
- Use `"auto"` model alias for hardware-optimized selection
