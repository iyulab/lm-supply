# LocalAI.Segmenter

Local image segmentation for .NET with automatic model downloading.

## Features

- **Zero-config**: Models download automatically from HuggingFace
- **GPU Acceleration**: CUDA, DirectML (Windows), CoreML (macOS)
- **MIT Licensed**: SegFormer models for commercial use
- **150 ADE20K Classes**: Indoor/outdoor scene understanding

## Quick Start

```csharp
using LocalAI.Segmenter;

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
| `interactive` | MobileSAM | ~40MB | - | Point/box prompts |

## GPU Acceleration

```bash
# NVIDIA GPU
dotnet add package Microsoft.ML.OnnxRuntime.Gpu

# Windows (AMD/Intel/NVIDIA)
dotnet add package Microsoft.ML.OnnxRuntime.DirectML
```
