# LMSupply.Detector

Local object detection for .NET with automatic model downloading.

## Features

- **Zero-config**: Models download automatically from HuggingFace
- **GPU Acceleration**: CUDA, DirectML (Windows), CoreML (macOS)
- **Permissively licensed**: RT-DETR (Apache-2.0) and YuNet (MIT), redistributable in a closed-source
  commercial product. No alias resolves to an AGPL-3.0 YOLO checkpoint.
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

| Alias | Model | Size | mAP | Licence | Classes |
|-------|-------|------|-----|---------|---------|
| `default` | RT-DETR v2 Small | ~80 MB | 48.1 | Apache-2.0 | COCO-80 |
| `fast` | RT-DETR v2 Mini-Small | ~126 MB | 46.0 | Apache-2.0 | COCO-80 |
| `quality` | RT-DETR v2 Medium | ~133 MB | 51.9 | Apache-2.0 | COCO-80 |
| `large` | RT-DETR v2 Large | ~169 MB | 53.4 | Apache-2.0 | COCO-80 |
| `xlarge` | RT-DETR v2 XLarge | ~300 MB | 54.3 | Apache-2.0 | COCO-80 |
| `face` | YuNet 2023mar | ~227 KB | n/a | MIT | `face`, with 5 landmarks |
| `plate` | LPD-YuNet 2023mar | ~4.1 MB | n/a | Apache-2.0 | `plate`, with 4 corners |

### Faces

`face` resolves to OpenCV's YuNet. COCO has no face class and `person` is not a substitute for redaction
work, so this is a separate model with its own single-label vocabulary and its own decoder.

```csharp
await using var detector = await LocalDetector.LoadAsync("face");

foreach (var face in await detector.DetectAsync("photo.jpg"))
{
    // face.Label is "face"; face.Keypoints holds the two eyes, the nose tip and the two mouth corners,
    // each carrying the detection's own score - YuNet publishes no per-landmark confidence.
    Console.WriteLine($"{face.Box.X1:F0},{face.Box.Y1:F0} {face.Box.Width:F0}x{face.Box.Height:F0}");
}
```

Feed it ordinary images: it wants BGR bytes rather than the scaled RGB the RT-DETR aliases take, and that
conversion happens inside the library.

**Measured cost** (1280x1177 JPEG, 4-core CPU, no GPU): about **32 ms per frame** end to end, of which
roughly half is JPEG decoding - the model itself runs in about 2.4 ms. Detection is therefore comfortably
inside a 30 fps budget when frames arrive already decoded, and JPEG decoding is the thing to avoid paying
twice for. DirectML rejects one of YuNet's operators, so it runs on CPU even on a machine where the RT-DETR
aliases get a GPU; the provider fallback handles this without configuration.

The defaults suit it: `ConfidenceThreshold` 0.25 and `IouThreshold` 0.45. A stricter `IouThreshold` of 0.3
matches the reference implementation and merges one more duplicate in a dense crowd; the difference measured
on a street scene was one box out of eight.

### Licence plates

`plate` resolves to OpenCV's licence-plate YuNet. It shares a name with the face model and little else -
320x240 input, prior boxes rather than an anchor-free grid, and a **quadrilateral** rather than an upright
box, because a plate photographed from an angle is not axis-aligned.

```csharp
await using var detector = await LocalDetector.LoadAsync("plate");

foreach (var plate in await detector.DetectAsync("photo.jpg"))
{
    // plate.Box is the upright hull - what you want if you are blurring the region.
    // plate.Keypoints holds the four corners, clockwise from top-left - what you want if you are
    // rectifying the plate to read it, or blurring a tighter polygon.
}
```

**Measured cost** (960x631 JPEG, 4-core CPU, no GPU): about **41 ms per frame** end to end; the model itself
runs in 5.3 ms. Like `face`, it runs on CPU rather than DirectML.

The library defaults (`ConfidenceThreshold` 0.25, `IouThreshold` 0.45) are usable: measured plates scored
0.63-0.99 while a cat photograph and a crowded street scene both produced nothing at all, the highest score
anywhere in them being 0.15. The reference implementation uses a stricter 0.8 with an IoU of 0.3; raise the
threshold if false positives cost you more than misses.


## GPU Acceleration

```bash
# NVIDIA GPU
dotnet add package Microsoft.ML.OnnxRuntime.Gpu

# Windows (AMD/Intel/NVIDIA)
dotnet add package Microsoft.ML.OnnxRuntime.DirectML
```
