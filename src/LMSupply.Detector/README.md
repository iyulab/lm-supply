# LMSupply.Detector

Local object detection for .NET with automatic model downloading.

## Features

- **Zero-config**: Models download automatically from HuggingFace
- **GPU Acceleration**: CUDA, DirectML (Windows), CoreML (macOS)
- **Permissively licensed**: RT-DETR (Apache-2.0), YuNet (MIT) and MoveNet (Apache-2.0), redistributable in
  a closed-source commercial product. No alias resolves to an AGPL-3.0 YOLO checkpoint.
- **Human pose**: `pose`, `pose-quality` and `pose-multi` return people with the 17 COCO keypoints
  (`DetectionResult.Keypoints`, indexed by `PoseSkeleton`).
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
| `pose` | MoveNet SinglePose Lightning | ~9.4 MB | n/a | Apache-2.0 | `person`, with 17 COCO keypoints |
| `pose-quality` | MoveNet SinglePose Thunder | ~25 MB | n/a | Apache-2.0 | `person`, with 17 COCO keypoints |
| `pose-multi` | MoveNet MultiPose Lightning | ~19 MB | n/a | Apache-2.0 | up to 6 `person`s, with 17 COCO keypoints |

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

**Measured cost** (1280x1177 JPEG, 4-core CPU): about **19 ms per frame** end to end, of which roughly half
is JPEG decoding - the model itself runs in about 2.4 ms. Detection is therefore comfortably inside a 30 fps
budget, and JPEG decoding is the thing to avoid paying twice for if frames arrive already decoded.
DirectML rejects one of this model's operators, so it runs on CPU even on a machine where the RT-DETR
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

**Measured cost** (960x631 JPEG, DirectML on an integrated GPU): about **7 ms per frame** end to end; on CPU
the model alone is 5.3 ms. Unlike `face`, this one does run on DirectML.

The library defaults (`ConfidenceThreshold` 0.25, `IouThreshold` 0.45) are usable: measured plates scored
0.63-0.99 while a cat photograph and a crowded street scene both produced nothing at all, the highest score
anywhere in them being 0.15. The reference implementation uses a stricter 0.8 with an IoU of 0.3; raise the
threshold if false positives cost you more than misses.

### Human pose

`pose`, `pose-quality` and `pose-multi` resolve to Google's MoveNet. Each person comes back as a `person`
detection whose `Keypoints` holds the 17 COCO keypoints in COCO order, so `PoseSkeleton` indices address
them directly, and every keypoint carries its own score.

```csharp
await using var detector = await LocalDetector.LoadAsync("pose");

foreach (var person in await detector.DetectAsync("photo.jpg"))
{
    var shoulder = person.Keypoints![PoseSkeleton.LeftShoulder];
    var elbow = person.Keypoints[PoseSkeleton.LeftElbow];
    if (shoulder.IsVisible(0.3f) && elbow.IsVisible(0.3f))
        Console.WriteLine($"upper arm: ({shoulder.X:F0},{shoulder.Y:F0}) -> ({elbow.X:F0},{elbow.Y:F0})");
}
```

- `pose` (SinglePose Lightning, 192x192) and `pose-quality` (SinglePose Thunder, 256x256) follow **one**
  person per frame - the most prominent one. They emit no person score and no box of their own, so the
  detection's `Confidence` is the mean of the 17 keypoint scores (a frame with nobody in it stays under the
  default `ConfidenceThreshold` of 0.25), and its `Box` is the extent of the keypoints - tighter than the
  person, since the top of the head and the hands lie outside it.
- `pose-multi` (MultiPose Lightning, 256x256) returns up to **six** people, each with the model's own box
  and score.
- Coordinates are pixels in the original image. The image is fed to the model with its aspect ratio kept
  and padded to a square, as MoveNet's reference preprocessing does; keypoints are not clamped to the image.

**Measured cost** (800x533 JPEG, a laptop-class desktop CPU with `ThreadCount = 4`, best of repeated runs):
about **5 ms** per frame end to end for `pose`, **10 ms** for `pose-quality` and **24 ms** for `pose-multi`,
of which about 2 ms is JPEG decoding. On a machine busy with other work, capping `ThreadCount` matters more
than the model choice: the default of one thread per core measured several times slower under contention.

**Licence.** The MoveNet weights are released by Google under Apache-2.0; the ONNX files are a community
conversion of them, published under Apache-2.0 (repository revisions are recorded in `DefaultModels`).
Google trained MoveNet on COCO and on its own internal dataset. The registry names the float32 build of each
repository; the int8 builds published beside it are not used, because on a test photograph they placed the
keypoints off the person. Pose models trained on datasets with non-commercial terms - AI Challenger,
CrowdPose, Halpe and the "body7" mixtures that include them, which covers the official RTMO/RTMPose ONNX
releases and ViTPose - are deliberately not offered.


## GPU Acceleration

Do not add ONNX Runtime packages (`Microsoft.ML.OnnxRuntime*`): LMSupply provisions the runtime itself, and a
second copy conflicts with it. `ExecutionProvider.Auto` uses CUDA when the CUDA 12 runtime and cuDNN 9 are
installed on the machine, CoreML on macOS, and the CPU otherwise. On Windows with an AMD or Intel GPU, ONNX
sessions run on the CPU (DirectML was removed in 0.67.0). See
[GPU acceleration](https://github.com/iyulab/lm-supply#gpu-acceleration).
