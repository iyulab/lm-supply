# LMSupply.Segmenter

Local image segmentation for .NET with automatic model downloading.

## Features

- **Zero-config**: Models download automatically from HuggingFace
- **GPU Acceleration**: CUDA, DirectML (Windows), CoreML (macOS)
- **MIT Licensed**: SegFormer models for commercial use
- **150 ADE20K Classes**: Indoor/outdoor scene understanding

## Quick Start

```csharp
using LMSupply.Segmenter;

// Load the default model
await using var segmenter = await LocalSegmenter.LoadAsync("default");

// Segment image
var result = await segmenter.SegmentAsync("photo.jpg");

Console.WriteLine($"Image: {result.Width}x{result.Height}");
Console.WriteLine($"Classes found: {result.UniqueClassCount}");

// Get class at pixel
int classId = result.GetClassAt(100, 100);
Console.WriteLine($"Class at (100,100): {segmenter.ClassLabels[classId]}");
```

## Available Models

| Alias | Model | Size | mIoU | Description |
|-------|-------|------|------|-------------|
| `default` | SegFormer-B0 | ~15MB | 38.0 | Lightweight, fast |
| `fast` | SegFormer-B1 | ~55MB | 42.2 | Balanced |
| `quality` | SegFormer-B2 | ~110MB | 46.5 | Higher accuracy |
| `large` | SegFormer-B5 | ~340MB | 51.0 | Highest accuracy |
| `interactive` | MobileSAM | ~45MB | - | Point/box prompts — `LocalSegmenter.LoadInteractiveAsync()` |

## GPU Acceleration

Do not add ONNX Runtime packages (`Microsoft.ML.OnnxRuntime*`): LMSupply provisions the runtime itself, and a
second copy conflicts with it. `ExecutionProvider.Auto` uses CUDA when the CUDA 12 runtime and cuDNN 9 are
installed on the machine, CoreML on macOS, and the CPU otherwise. On Windows with an AMD or Intel GPU, ONNX
sessions run on the CPU (DirectML was removed in 0.67.0). See
[GPU acceleration](https://github.com/iyulab/lm-supply#gpu-acceleration).
