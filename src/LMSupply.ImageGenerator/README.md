# LMSupply.ImageGenerator

A simple .NET library for local text-to-image generation using Latent Consistency Models (LCM). Designed for edge deployment with fast 2-4 step inference.

## Features

- **Fast Generation**: LCM enables 2-4 step inference (vs 20-50 for standard diffusion)
- **Edge-Optimized**: ~3-4GB VRAM requirement, suitable for consumer GPUs
- **GPU Acceleration**: CUDA, DirectML, and CoreML support via ONNX Runtime
- **Lazy Downloads**: Models downloaded on-demand from HuggingFace
- **Streaming Progress**: Real-time step-by-step progress with optional previews

## Quick Start

```csharp
using LMSupply.ImageGenerator;

// Load the default LCM model
await using var generator = await LocalImageGenerator.LoadAsync("default");

// Generate an image
var result = await generator.GenerateAsync("A sunset over mountains");
await result.SaveAsync("output.png");
```

## Model Aliases

| Alias | Description | Steps | Use Case |
|-------|-------------|-------|----------|
| `default` | LCM-Dreamshaper-V7 | 4 | Best balance of quality and speed |
| `fast` | LCM-Dreamshaper-V7 | 2 | Fastest generation, slightly lower quality |
| `quality` | LCM-Dreamshaper-V7 | 4 | Higher guidance scale for better adherence |

## Generation Options

```csharp
var options = new GenerationOptions
{
    Width = 512,           // Image width (must be divisible by 8)
    Height = 512,          // Image height (must be divisible by 8)
    Steps = 4,             // Inference steps (2-4 for LCM)
    GuidanceScale = 1.0f,  // CFG scale (1.0-2.0 for LCM)
    Seed = 42,             // Random seed for reproducibility
    NegativePrompt = "blurry, low quality",
    GeneratePreviews = true // Enable step-by-step previews
};

var result = await generator.GenerateAsync("A cat wearing a hat", options);
```

## Streaming Generation

```csharp
await foreach (var step in generator.GenerateStreamingAsync("A beautiful landscape"))
{
    Console.WriteLine($"Step {step.StepNumber}/{step.TotalSteps} ({step.Progress:F1}%)");

    if (step.HasPreview)
    {
        // Save intermediate preview
        await File.WriteAllBytesAsync($"preview_{step.StepNumber}.png", step.PreviewData!);
    }

    if (step.IsFinal && step.FinalImage != null)
    {
        await step.FinalImage.SaveAsync("final.png");
    }
}
```

## Batch Generation

```csharp
// Generate 4 variations with different seeds
var images = await generator.GenerateBatchAsync("A fantasy castle", count: 4);

for (int i = 0; i < images.Length; i++)
{
    await images[i].SaveAsync($"castle_{i}.png");
    Console.WriteLine($"Image {i}: Seed={images[i].Seed}, Time={images[i].GenerationTime.TotalSeconds:F2}s");
}
```

## GPU Configuration

```csharp
using LMSupply;   // ExecutionProvider

var options = new ImageGeneratorOptions
{
    Provider = ExecutionProvider.Cuda,  // Force CUDA (auto-detected by default)
    DeviceId = 0,                       // GPU device index
    UseFp16 = true,                     // Use FP16 for reduced memory
    CacheDirectory = "/path/to/cache"   // Custom model cache location
};

await using var generator = await LocalImageGenerator.LoadAsync("default", options);
```

## Requirements

- .NET 10.0+
- GPU with 3-4GB VRAM (recommended)
- Supported execution providers:
  - CUDA 11/12 (NVIDIA GPUs)
  - DirectML (Windows, any DirectX 12 GPU)
  - CoreML (macOS, Apple Silicon)
  - CPU (fallback, slower)

## How It Works

LMSupply.ImageGenerator uses **Latent Consistency Models (LCM)**, a distillation technique that enables high-quality image generation in just 2-4 steps instead of the traditional 20-50 steps required by Stable Diffusion.

The pipeline consists of:
1. **CLIP Text Encoder**: Converts text prompts to embeddings
2. **LCM UNet**: Denoises latents in 2-4 steps
3. **VAE Decoder**: Converts latents to pixel space
4. **LCM Scheduler**: Manages the accelerated diffusion process

## License

MIT License - See LICENSE file for details.
