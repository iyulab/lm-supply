# LMSupply.Captioner

Local image captioning for .NET with automatic model downloading.

## Features

- **Zero-config**: Models download automatically from HuggingFace
- **GPU Acceleration**: CUDA, DirectML (Windows), CoreML (macOS)
- **Cross-platform**: Windows, Linux, macOS
- **Simple API**: Just 2 lines of code to get started

## Quick Start

```csharp
using LMSupply.Captioner;

// Load the default captioning model
var captioner = await LocalCaptioner.LoadAsync("default");

// Generate a caption
var result = await captioner.CaptionAsync("photo.jpg");
Console.WriteLine(result.Caption);
// Output: "A cat sitting on a windowsill looking outside"
```

## Available Models

| Model ID | Size | Description |
|----------|------|-------------|
| `default` | ~500MB | ViT-GPT2 - fast, general purpose |
| `vit-gpt2` | ~500MB | Same as default |
| `smolvlm` | ~600MB | SmolVLM-256M - lightweight multimodal |
| `florence2` | ~500MB | Florence-2-base - multi-task vision |

## Advanced Usage

```csharp
// Custom options
var options = new CaptionerOptions
{
    MaxLength = 50,
    Provider = ExecutionProvider.DirectML
};

var captioner = await LocalCaptioner.LoadAsync("default", options);
var result = await captioner.CaptionAsync("image.jpg");

Console.WriteLine($"Caption: {result.Caption}");
Console.WriteLine($"Confidence: {result.Confidence:P1}");
```

## GPU Acceleration

Install the appropriate GPU package for your hardware:

```bash
# NVIDIA GPU
dotnet add package Microsoft.ML.OnnxRuntime.Gpu

# Windows (AMD/Intel/NVIDIA)
dotnet add package Microsoft.ML.OnnxRuntime.DirectML
```
