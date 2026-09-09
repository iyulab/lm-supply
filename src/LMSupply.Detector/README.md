# LMSupply.Detector

Local object detection for .NET with automatic model downloading.

## Features

- **Zero-config**: Models download automatically from HuggingFace
- **GPU Acceleration**: CUDA, DirectML (Windows), CoreML (macOS)
- **Apache-2.0 Licensed**: RT-DETR models for commercial use
- **Per-model vocabulary**: COCO-80 (people, vehicles, animals, objects) by default, and a model that was
  trained on something else carries its own labels — `DetectorModelInfo.ClassLabels`, with `NumClasses`
  derived from it so the two cannot disagree. Post-processing labels from the loaded model, not from COCO
  by assumption; an id outside the vocabulary reads as `unknown` rather than borrowing a COCO name.

## Quick Start

```csharp
using LMSupply.Detector;

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
