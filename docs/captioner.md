# LMSupply.Captioner

A lightweight, zero-configuration image captioning library for .NET with automatic GPU acceleration.

## Installation

```bash
dotnet add package LMSupply.Captioner
```

For GPU acceleration:

```bash
# NVIDIA CUDA
dotnet add package Microsoft.ML.OnnxRuntime.Gpu

# Windows DirectML
dotnet add package Microsoft.ML.OnnxRuntime.DirectML

# macOS CoreML
dotnet add package Microsoft.ML.OnnxRuntime.CoreML
```

## Basic Usage

```csharp
using LMSupply.Captioner;

// Load the default model
await using var captioner = await LocalCaptioner.LoadAsync("default");

// Generate a caption from file
var result = await captioner.CaptionAsync("photo.jpg");
Console.WriteLine(result.Caption);
// Output: "A cat sitting on a windowsill looking outside"

// Generate a caption from stream
using var stream = File.OpenRead("image.png");
var result = await captioner.CaptionAsync(stream);
```

## Available Models

| Alias | Model | Size | Description |
|-------|-------|------|-------------|
| `default` | ViT-GPT2 | ~500MB | Fast, general purpose captioning |
| `vit-gpt2` | ViT-GPT2 | ~500MB | Same as default |
| `smolvlm` | SmolVLM-256M | ~600MB | Lightweight multimodal model |
| `florence2` | Florence-2-base | ~500MB | Multi-task vision model |

## Advanced Usage

### Custom Options

```csharp
var options = new CaptionerOptions
{
    MaxLength = 50,                        // Maximum caption length
    Provider = ExecutionProvider.DirectML, // Force specific GPU provider
    CacheDirectory = "/custom/cache"       // Custom model cache directory
};

var captioner = await LocalCaptioner.LoadAsync("default", options);
var result = await captioner.CaptionAsync("image.jpg");

Console.WriteLine($"Caption: {result.Caption}");
Console.WriteLine($"Confidence: {result.Confidence:P1}");
Console.WriteLine($"Processing time: {result.ProcessingTimeMs}ms");
```

### Batch Processing

```csharp
var images = new[] { "image1.jpg", "image2.jpg", "image3.jpg" };

foreach (var image in images)
{
    var result = await captioner.CaptionAsync(image);
    Console.WriteLine($"{image}: {result.Caption}");
}
```

### Using Byte Arrays

```csharp
// Caption from byte array (useful for API scenarios)
byte[] imageBytes = await httpClient.GetByteArrayAsync(imageUrl);
var result = await captioner.CaptionAsync(imageBytes);
```

## GPU Acceleration

GPU acceleration is automatic when available. Priority order:
1. CUDA (NVIDIA GPUs)
2. DirectML (Windows - AMD, Intel, NVIDIA)
3. CoreML (macOS)
4. CPU (fallback)

Force a specific provider:

```csharp
var options = new CaptionerOptions
{
    Provider = ExecutionProvider.Cuda
};
```

## Model Caching

Models are cached following HuggingFace Hub conventions:
- Default: `~/.cache/huggingface/hub`
- Override via: `HF_HUB_CACHE`, `HF_HOME`, or `XDG_CACHE_HOME` environment variables
- Or set `CaptionerOptions.CacheDirectory`
