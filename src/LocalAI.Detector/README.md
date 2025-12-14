# LocalAI.Detector

Local object detection for .NET with automatic model downloading.

## Features

- **Zero-config**: Models download automatically from HuggingFace
- **GPU Acceleration**: CUDA, DirectML (Windows), CoreML (macOS)
- **Apache-2.0 Licensed**: RT-DETR models for commercial use
- **80 COCO Classes**: People, vehicles, animals, objects

## Quick Start

```csharp
using LocalAI.Detector;

// Load the default model
await using var detector = await LocalDetector.LoadAsync("default");

// Detect objects
var results = await detector.DetectAsync("photo.jpg");

foreach (var detection in results)
{
    Console.WriteLine($"{detection.Label}: {detection.Confidence:P1}");
    Console.WriteLine($"  Box: [{detection.Box.X1:F0}, {detection.Box.Y1:F0}]");
}
```

## Available Models

| Alias | Model | Size | mAP | Description |
|-------|-------|------|-----|-------------|
| `default` | RT-DETR R18 | ~80MB | 46.5 | Best balance |
| `fast` | EfficientDet-D0 | ~15MB | 33.8 | Fastest inference |
| `quality` | RT-DETR R50 | ~170MB | 53.1 | Higher accuracy |
| `large` | RT-DETR R101 | ~300MB | 54.3 | Highest accuracy |

## GPU Acceleration

```bash
# NVIDIA GPU
dotnet add package Microsoft.ML.OnnxRuntime.Gpu

# Windows (AMD/Intel/NVIDIA)
dotnet add package Microsoft.ML.OnnxRuntime.DirectML
```
