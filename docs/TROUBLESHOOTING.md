# Troubleshooting Guide

Common issues and solutions when using LMSupply.

---

## 1. Installation Issues

### 1.1 Package Restore Fails

**Symptom:** NuGet restore fails with version conflicts.

**Solution:**
```bash
# Clear NuGet cache
dotnet nuget locals all --clear

# Restore again
dotnet restore
```

### 1.2 ONNX Runtime Conflicts

**Symptom:** DLL conflicts or "assembly not found" errors.

**Cause:** Multiple ONNX Runtime packages installed.

**Solution:**
```bash
# Remove conflicting packages
dotnet remove package Microsoft.ML.OnnxRuntime
dotnet remove package Microsoft.ML.OnnxRuntime.Gpu
dotnet remove package Microsoft.ML.OnnxRuntime.DirectML

# LMSupply manages ONNX Runtime automatically
```

---

## 2. Model Loading Issues

### 2.1 ModelNotFoundException

**Symptom:**
```
ModelNotFoundException: Model 'invalid-model' not found
```

**Causes:**
- Typo in model ID
- Model not in registry
- Invalid HuggingFace ID

**Solution:**
```csharp
// Use built-in aliases
await LocalEmbedder.LoadAsync("default");  // or "fast", "quality", "auto"

// Or use full HuggingFace ID
await LocalEmbedder.LoadAsync("BAAI/bge-base-en-v1.5");
```

### 2.2 Download Failures

**Symptom:** Model download hangs or fails.

**Causes:**
- Network issues
- HuggingFace Hub unavailable
- Firewall blocking

**Solutions:**
```csharp
// 1. Use progress callback to monitor
var progress = new Progress<DownloadProgress>(p =>
    Console.WriteLine($"{p.OverallPercentComplete:F0}%"));

await LocalEmbedder.LoadAsync("default", progress: progress);

// 2. Set custom timeout via HttpClient
// (Advanced: Configure in custom cache directory)

// 3. Pre-download models
// Models are cached at: ~/.cache/lm-supply/
```

### 2.3 ModelLoadException

**Symptom:**
```
ModelLoadException: Failed to load model
```

**Causes:**
- Corrupt model files
- Incompatible ONNX version
- Missing dependencies

**Solutions:**
```bash
# 1. Clear cache and re-download
rm -rf ~/.cache/lm-supply/

# 2. Check disk space
df -h ~/.cache/

# 3. Verify ONNX Runtime compatibility
```

---

## 3. GPU Issues

### 3.1 CUDA Not Detected

**Symptom:** Model runs on CPU despite NVIDIA GPU.

**Diagnostic:**
```csharp
var profile = HardwareProfile.Current;
Console.WriteLine($"GPU: {profile.GpuInfo.DeviceName}");
Console.WriteLine($"Vendor: {profile.GpuInfo.Vendor}");
Console.WriteLine($"VRAM: {profile.GpuMemoryGB:F1} GB");
Console.WriteLine($"Provider: {profile.RecommendedProvider}");
```

**Causes & Solutions:**

| Cause | Solution |
|-------|----------|
| Outdated driver | Update NVIDIA driver to 450+ |
| CUDA not installed | CUDA toolkit not required for inference |
| Low VRAM | Need 4GB+ for CUDA selection |
| WSL2 | Ensure CUDA WSL support is enabled |

### 3.2 DirectML Errors

**Symptom:** DirectML initialization fails on Windows.

**Solutions:**
1. Update GPU driver
2. Ensure Windows 10 1903+ or Windows 11
3. Check DirectX 12 support:
```powershell
dxdiag
# Check "Feature Levels" includes 12_0
```

### 3.3 Out of Memory (OOM)

**Symptom:**
```
OutOfMemoryException or CUDA out of memory
```

**Solutions:**
```csharp
// 1. Use "auto" for hardware-appropriate model
await LocalEmbedder.LoadAsync("auto");

// 2. Use smaller model explicitly
await LocalEmbedder.LoadAsync("fast");

// 3. Load models sequentially, not simultaneously
await using (var embedder = await LocalEmbedder.LoadAsync("default"))
{
    // Use embedder
}
// VRAM freed before loading next model

await using (var generator = await LocalGenerator.LoadAsync("auto"))
{
    // Use generator
}

// 4. Force CPU for large models
var options = new GeneratorOptions { Provider = ExecutionProvider.Cpu };
await LocalGenerator.LoadAsync("auto", options);
```

---

## 4. Inference Issues

### 4.1 Slow First Inference

**Symptom:** First call is much slower than subsequent calls.

**Cause:** JIT compilation and GPU kernel initialization.

**Solution:**
```csharp
await using var model = await LocalEmbedder.LoadAsync("default");

// Warmup before production use
await model.WarmupAsync();

// Now inference is fast
var result = await model.EmbedAsync("text");
```

### 4.2 InferenceException

**Symptom:**
```
InferenceException: Runtime inference error
```

**Causes:**
- Invalid input
- Model corruption
- Resource exhaustion

**Solutions:**
```csharp
// 1. Validate input
if (string.IsNullOrWhiteSpace(text))
    throw new ArgumentException("Text cannot be empty");

// 2. Check input length
var info = model.GetModelInfo();
if (text.Length > info.MaxSequenceLength * 4)  // Rough estimate
    Console.WriteLine("Warning: Text may be truncated");

// 3. Implement retry with backoff
var retryCount = 0;
while (retryCount < 3)
{
    try
    {
        return await model.EmbedAsync(text);
    }
    catch (InferenceException) when (retryCount < 2)
    {
        retryCount++;
        await Task.Delay(100 * retryCount);
    }
}
```

---

## 5. Platform-Specific Issues

### 5.1 Linux

**CUDA on Linux:**
```bash
# Check NVIDIA driver
nvidia-smi

# Verify CUDA compatibility
# Driver 450+ required
```

**Library loading issues:**
```bash
# Add to LD_LIBRARY_PATH if needed
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 5.2 macOS

**Apple Silicon (M1/M2/M3):**
- CoreML is used automatically
- Ensure macOS 11.0+
- Rosetta 2 may affect performance for x64 binaries

**Intel Mac:**
- CPU is primary provider
- AMD GPU may use CoreML

### 5.3 Windows

**Long path issues:**
```powershell
# Enable long paths (requires admin)
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v "LongPathsEnabled" /t REG_DWORD /d 1 /f
```

**Antivirus interference:**
- Add cache directory to exclusions: `%USERPROFILE%\.cache\lm-supply`

---

## 6. Common Error Messages

| Error | Likely Cause | Quick Fix |
|-------|--------------|-----------|
| `ModelNotFoundException` | Invalid model ID | Use built-in alias |
| `ModelLoadException` | Corrupt/missing files | Clear cache |
| `InferenceException` | Runtime error | Check input, retry |
| `OutOfMemoryException` | Insufficient VRAM | Use smaller model |
| `DllNotFoundException` | Missing native lib | Reinstall package |
| `TimeoutException` | Network issue | Check connectivity |

---

## 7. Diagnostic Checklist

When reporting issues, collect this information:

```csharp
// Hardware info
var profile = HardwareProfile.Current;
Console.WriteLine($"OS: {Environment.OSVersion}");
Console.WriteLine($".NET: {Environment.Version}");
Console.WriteLine($"GPU: {profile.GpuInfo.DeviceName}");
Console.WriteLine($"GPU Vendor: {profile.GpuInfo.Vendor}");
Console.WriteLine($"GPU Memory: {profile.GpuMemoryGB:F1} GB");
Console.WriteLine($"System Memory: {profile.SystemMemoryGB:F1} GB");
Console.WriteLine($"Provider: {profile.RecommendedProvider}");
Console.WriteLine($"Tier: {profile.Tier}");
```

---

## 8. Getting Help

1. **Check this guide** for common solutions
2. **Search GitHub Issues**: [LMSupply Issues](https://github.com/iyulab/lm-supply/issues)
3. **Create new issue** with:
   - LMSupply version
   - .NET version
   - OS and hardware info
   - Minimal reproduction code
   - Full error message and stack trace
