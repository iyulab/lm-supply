# LMSupply.Segmenter

Local image segmentation for .NET with automatic model downloading.

## Features

- **Zero-config**: Models download automatically from HuggingFace
- **GPU Acceleration**: CUDA (NVIDIA), CoreML (macOS)
- **Permissive licences**: the built-in models are Apache-2.0 or MIT
- **ADE20K Classes**: 150 (or the full 847) classes of indoor/outdoor scene understanding; a portrait model that finds people

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

| Alias | Model | Size | mIoU | Classes | Description |
|-------|-------|------|------|---------|-------------|
| `default` | SegFormer-B0 | ~15MB | 38.0 | 150 (ADE20K) | Lightweight, fast |
| `fast` | MediaPipe Selfie | ~3MB | - | 2 (`background`, `person`) | Portrait and selfie segmentation: finds people, nothing else |
| `quality` | MaskFormer ResNet50 | ~178MB | 44.5 | 847 (full ADE20K) | Higher accuracy over the full label set |
| `large` | SegFormer-B0 | ~15MB | 38.0 | 150 (ADE20K) | The same model as `default` |
| `interactive` | MobileSAM | ~45MB | - | - | Point/box prompts — `LocalSegmenter.LoadInteractiveAsync()` |

`ClassLabels` names the classes of the model that is loaded, in class-id order.

## GPU Acceleration

Do not add ONNX Runtime packages (`Microsoft.ML.OnnxRuntime*`): LMSupply provisions the runtime itself, and a
second copy conflicts with it. `ExecutionProvider.Auto` uses CUDA when the driver supports CUDA 13 and the CUDA 13
runtime and cuDNN 9 are installed on the machine (ONNX Runtime 1.30 is built for CUDA 13), CoreML on macOS, and the CPU otherwise. On Windows with an AMD or Intel GPU, ONNX
sessions run on the CPU (DirectML was removed in 0.67.0). See
[GPU acceleration](https://github.com/iyulab/lm-supply#gpu-acceleration).
