# GPU Providers Guide

This guide explains execution providers in LMSupply and how to optimize GPU acceleration.

## Overview

LMSupply uses ONNX Runtime for inference, which supports multiple execution providers for hardware acceleration.

---

## 1. Available Providers

| Provider | Platform | GPU Vendor | Notes |
|----------|----------|------------|-------|
| **CUDA** | Windows/Linux | NVIDIA | Best performance for NVIDIA GPUs |
| **DirectML** | Windows | Any (NVIDIA/AMD/Intel) | Universal Windows GPU support |
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
2. **DirectML** - If Windows with compatible GPU
3. **CoreML** - If macOS with Apple Silicon
4. **CPU** - Fallback

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
- `ExecutionProvider.DirectML`
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

### 3.2 DirectML (Windows)

**Requirements:**
- Windows 10 version 1903+
- DirectX 12 compatible GPU
- Updated GPU drivers

**Performance Characteristics:**
- Good performance across vendors
- Slightly slower than CUDA for NVIDIA
- Best option for AMD GPUs on Windows

**When to use:**
- AMD GPU on Windows
- Intel integrated/discrete GPU
- NVIDIA without CUDA toolkit

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
1. CUDA → DirectML → CPU (Windows)
2. CUDA → CPU (Linux)
3. CoreML → CPU (macOS)

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
`[<Module>] Inference failed on DirectML (...)` / `[<Module>] Inference timed out on DirectML after 60s ...`
followed by `[<Module>] Recovered: now running on CPUExecutionProvider.` Attach a
`TraceListener` (see `samples/EmbedderSample`) if you want to see them. After a recovery,
`IsGpuActive` / `ActiveProviders` on the model reflect the provider actually in use.

A model made of several sessions (an encoder + decoder pair, for example) shares one provider
blacklist across them: once one session has left a provider — at run time, or because the load-time
chain already saw that provider fail for this model — its siblings leave it before their next run
instead of hitting the same crash or hang themselves. For an autoregressive decoder the bound and the
recovery apply **per decode step**, so a long output is never cut off by a whole-loop timeout.

Modules on this recovery path: Embedder, Reranker, Segmenter (SegFormer and MobileSAM), Ocr (detection
and recognition), Synthesizer, Detector, Transcriber, Translator, Captioner (each encoder + decoder). The
remaining ONNX modules are being migrated; until then they run under the inference bound only
(a hang surfaces as `InferenceTimeoutException` without a provider switch).

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
    !info.ActiveProviders.Contains("DmlExecutionProvider"))
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

| Aspect | CUDA | DirectML | CoreML | CPU |
|--------|------|----------|--------|-----|
| **Speed** | Fastest | Fast | Fast | Slowest |
| **Latency** | Lowest | Low | Low | Highest |
| **Batch Perf** | Excellent | Good | Good | Moderate |
| **Memory** | GPU VRAM | GPU VRAM | Unified | System RAM |
| **Setup** | Driver only | Auto | Auto | None |

---

## 8. Summary

- **Auto-detection** handles most cases correctly
- **CUDA** is best for NVIDIA GPUs
- **DirectML** is the universal Windows option
- **CoreML** is optimal for Apple Silicon
- **CPU** is always available as fallback
- Use `HardwareProfile.Current` to check detected hardware
- Use `"auto"` model alias for hardware-optimized selection
