# LMSupply.Captioner

Local image captioning for .NET with automatic model downloading.

## Features

- **Zero-config**: Models download automatically from HuggingFace
- **GPU Acceleration**: CUDA, CoreML (macOS)
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

| Model ID | Download | Description |
|----------|----------|-------------|
| `default` (also `fast`, `auto`) | ~1 GB | ViT-GPT2 — fast, one-sentence captions; tends to name a familiar scene rather than the subject |
| `quality` | ~275 MB (int8) · ~545 MB (fp16) · ~1.1 GB (fp32) | Florence-2 base — names the main subjects of everyday photos; brief, detailed or paragraph captions |

`quality` is published in quantized variants; a load takes the one the hardware tier picks, or the one you name with
`QuantizationHint` or a qualifier (`"quality:fp16"`). `LocalCaptioner.GetDownloadSizeBytesAsync` answers what a load
will download, for a consent screen.

```csharp
using LMSupply.Captioner;

long bytes = await LocalCaptioner.GetDownloadSizeBytesAsync("quality");

await using var captioner = await LocalCaptioner.LoadAsync("quality",
    new CaptionerOptions { Detail = CaptionDetail.Detailed, MaxLength = 100 });
var result = await captioner.CaptionAsync("photo.jpg");
```

`Detail` (`Brief` · `Detailed` · `Paragraph`) is read by `quality`; `default` captions at one level and refuses any other
value at load. `Prompt` (caption prefix) is read by `default`; `quality` refuses it — its prompt selects the task.

## Advanced Usage

```csharp
using LMSupply;   // ExecutionProvider

// Custom options
var options = new CaptionerOptions
{
    MaxLength = 50,
    Provider = ExecutionProvider.Cuda
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
```

AMD / Intel GPUs on Windows have no ONNX provider on ONNX Runtime 1.25+ (DirectML was removed in 0.67.0); captioning
runs on CPU there.
