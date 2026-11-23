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
| `entry` | ~1 GB | ViT-GPT2 — fast, one-sentence captions; tends to name a familiar scene rather than the subject |
| `default` (also `basic`, `florence-2-base`, `auto`) | ~275 MB (int8) · ~545 MB (fp16) · ~1.1 GB (fp32) | Florence-2 base — names the main subjects of everyday photos; brief, detailed or paragraph captions |

`basic` is published in quantized variants; a load takes the one the hardware tier picks, or the one you name with
`QuantizationHint` or a qualifier (`"quality:fp16"`). `LocalCaptioner.GetDownloadSizeBytesAsync` answers what a load
will download, for a consent screen.

```csharp
using LMSupply.Captioner;

long bytes = await LocalCaptioner.GetDownloadSizeBytesAsync("basic");

await using var captioner = await LocalCaptioner.LoadAsync("basic",
    new CaptionerOptions { Detail = CaptionDetail.Detailed, MaxLength = 100 });
var result = await captioner.CaptionAsync("photo.jpg");
```

`Detail` (`Brief` · `Detailed` · `Paragraph`) is read by `basic` (`default`); `entry` captions at one level and refuses any other
value at load. `Prompt` (caption prefix) is read by `entry`; `basic` refuses it — its prompt selects the task.

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

Do not add ONNX Runtime packages (`Microsoft.ML.OnnxRuntime*`): LMSupply provisions the runtime itself, and a
second copy conflicts with it. `ExecutionProvider.Auto` uses CUDA when the driver supports CUDA 13 and the CUDA 13
runtime and cuDNN 9 are installed on the machine (ONNX Runtime 1.30 is built for CUDA 13), CoreML on macOS, and the CPU otherwise. On Windows with an AMD or Intel GPU, ONNX
sessions run on the CPU (DirectML was removed in 0.67.0). See
[GPU acceleration](https://github.com/iyulab/lm-supply#gpu-acceleration).
