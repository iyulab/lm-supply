# LMSupply.Detector

A lightweight, zero-configuration object detection library for .NET with automatic GPU acceleration.

## Installation

```bash
dotnet add package LMSupply.Detector
```

For GPU acceleration:

```bash
# NVIDIA CUDA
dotnet add package Microsoft.ML.OnnxRuntime.Gpu

# Windows DirectML
dotnet add package Microsoft.ML.OnnxRuntime.DirectML

# macOS CoreML
dotnet add package Microsoft.ML.OnnxRuntime.CoreML
```

## Basic Usage

```csharp
using LMSupply.Detector;

// Load the default model
await using var detector = await LocalDetector.LoadAsync("default");

// Detect objects in an image
var results = await detector.DetectAsync("photo.jpg");

foreach (var detection in results)
{
    Console.WriteLine($"{detection.Label}: {detection.Confidence:P1}");
    Console.WriteLine($"  Box: [{detection.Box.X1:F0}, {detection.Box.Y1:F0}] - [{detection.Box.X2:F0}, {detection.Box.Y2:F0}]");
}
// Output:
// person: 95.2%
//   Box: [120, 50] - [380, 450]
// car: 87.3%
//   Box: [400, 200] - [600, 350]
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

Every alias is permissively licensed and redistributable in a closed-source commercial product. No alias
resolves to a YOLO checkpoint: those are AGPL-3.0 and would carry that obligation to the consumer.

The RT-DETR aliases are NMS-free and share the COCO-80 vocabulary. `face` is a different architecture with a
different vocabulary, so it is described separately below.

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


You can also use any HuggingFace object detection model by its full ID:

```csharp
// Use any ONNX detection model from HuggingFace
var detector = await LocalDetector.LoadAsync("PekingU/rtdetr_r18vd");
```

## Advanced Usage

### Custom Options

```csharp
var options = new DetectorOptions
{
    ConfidenceThreshold = 0.5f,            // Only return detections above 50%
    IouThreshold = 0.45f,                  // NMS IoU threshold (for non-RT-DETR models)
    MaxDetections = 50,                    // Maximum detections to return
    Provider = ExecutionProvider.DirectML, // Force specific GPU provider
    CacheDirectory = "/custom/cache"       // Custom model cache directory
};

var detector = await LocalDetector.LoadAsync("quality", options);
```

### Class Filtering

```csharp
// Only detect people and cars (COCO class IDs)
var options = new DetectorOptions
{
    ClassFilter = new HashSet<int> { 0, 2 } // 0=person, 2=car
};

var detector = await LocalDetector.LoadAsync("default", options);
var results = await detector.DetectAsync("street.jpg");
```

### Batch Processing

```csharp
var images = new[] { "image1.jpg", "image2.jpg", "image3.jpg" };
var batchResults = await detector.DetectBatchAsync(images);

for (int i = 0; i < images.Length; i++)
{
    Console.WriteLine($"{images[i]}: {batchResults[i].Count} objects detected");
}
```

### Using Streams and Byte Arrays

```csharp
// From stream
using var stream = File.OpenRead("image.png");
var results = await detector.DetectAsync(stream);

// From byte array (useful for API scenarios)
byte[] imageBytes = await httpClient.GetByteArrayAsync(imageUrl);
var results = await detector.DetectAsync(imageBytes);
```

### Working with Bounding Boxes

```csharp
var results = await detector.DetectAsync("photo.jpg");

foreach (var det in results)
{
    var box = det.Box;

    // Get box properties
    Console.WriteLine($"Width: {box.Width}, Height: {box.Height}");
    Console.WriteLine($"Center: ({box.CenterX}, {box.CenterY})");
    Console.WriteLine($"Area: {box.Area} pixels");

    // Scale to different dimensions
    var scaledBox = box.Scale(0.5f, 0.5f);

    // Clamp to image boundaries
    var clampedBox = box.Clamp(imageWidth, imageHeight);

    // Calculate IoU with another box
    float iou = box.IoU(otherBox);
}
```

### Accessing Class Labels

```csharp
// Get all COCO class labels
var labels = LocalDetector.CocoClassLabels;
Console.WriteLine(string.Join(", ", labels.Take(5)));
// Output: person, bicycle, car, motorcycle, airplane

// Or from the model instance
var modelLabels = detector.ClassLabels;
```

### Describing a model's tensor layout

A detector is not only a set of weights: it also fixes how the input tensor is built and how the raw
output is read. Both are declared per model rather than assumed, because getting either wrong is silent —
a model fed the opposite channel order raises nothing and returns an empty result, which is
indistinguishable from a photograph containing none of what was being looked for.

`DetectorInputFormat` names the input convention:

| Value | Channel order | Pixel values |
|-------|---------------|--------------|
| `ScaledRgb` | RGB | scaled to `0..1`, no mean/standard-deviation shift |
| `RawBgr` | BGR | raw `0..255`, no scaling and no shift |

`DetectorOutputLayout` names the head the decoder must read, and `RequiresNms` and `NumKeypoints` are
derived from it — so a description cannot contradict the decoder that acts on it.

Both are surfaced on `DetectorModelInfo`, and `DetectorOptions.InputFormat` can override the input
convention when loading a model the built-in registry does not describe.

## GPU Acceleration

GPU acceleration is automatic when available. Priority order:
1. CUDA (NVIDIA GPUs)
2. DirectML (Windows - AMD, Intel, NVIDIA)
3. CoreML (macOS)
4. CPU (fallback)

Force a specific provider:

```csharp
var options = new DetectorOptions
{
    Provider = ExecutionProvider.Cuda
};
```

## Model Caching

Models are cached following HuggingFace Hub conventions:
- Default: `~/.cache/huggingface/hub`
- Override via: `HF_HUB_CACHE`, `HF_HOME`, or `XDG_CACHE_HOME` environment variables
- Or set `DetectorOptions.CacheDirectory`

## Upgrading from 0.61

**RT-DETR scores move slightly, so a fixed threshold returns a different count.** Preprocessing used to
shift every model by the ImageNet mean and standard deviation. The RT-DETR reference preprocessing does
not do that — its own image processor ships those statistics with normalisation switched off — so the
RT-DETR aliases now scale to `0..1` and stop there.

The difference is small and runs in both directions, but it is visible at a fixed threshold. Measured on
three photographs:

| Input | 0.62 (aligned) | 0.61 |
|-------|----------------|------|
| single cat | `cat` 0.962 | `cat` 0.959 |
| single dog | `dog` 0.965 | `dog` 0.950 |
| crowd, threshold 0.5 | 17 detections, mean 0.705 | 21 detections, mean 0.651 |
| crowd, threshold 0.3 | 49 detections, mean 0.498 | 44 detections, mean 0.509 |

If you tuned `ConfidenceThreshold` against 0.61 output, re-check it. Nothing changes for `face` or
`plate`, which were introduced in this release.

**Two properties on `DetectorModelInfo` were renamed.**

| Before | After | Why |
|--------|-------|-----|
| `InputSize` | `InputWidth`, `InputHeight` | a single value cannot describe a model whose input is not square |
| `NumKeypoints` | `OutputLayout` | the keypoint count and the NMS answer are now derived from the layout, so the two cannot disagree |

## COCO Class Reference

The default models detect 80 COCO classes:

| ID | Class | ID | Class | ID | Class | ID | Class |
|----|-------|----|----|----|----|----|----|
| 0 | person | 20 | elephant | 40 | wine glass | 60 | dining table |
| 1 | bicycle | 21 | bear | 41 | cup | 61 | toilet |
| 2 | car | 22 | zebra | 42 | fork | 62 | tv |
| 3 | motorcycle | 23 | giraffe | 43 | knife | 63 | laptop |
| 4 | airplane | 24 | backpack | 44 | spoon | 64 | mouse |
| 5 | bus | 25 | umbrella | 45 | bowl | 65 | remote |
| 6 | train | 26 | handbag | 46 | banana | 66 | keyboard |
| 7 | truck | 27 | tie | 47 | apple | 67 | cell phone |
| 8 | boat | 28 | suitcase | 48 | sandwich | 68 | microwave |
| 9 | traffic light | 29 | frisbee | 49 | orange | 69 | oven |
