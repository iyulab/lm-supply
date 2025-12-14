# LocalAI.Detector

A lightweight, zero-configuration object detection library for .NET with automatic GPU acceleration.

## Installation

```bash
dotnet add package LocalAI.Detector
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
using LocalAI.Detector;

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

| Alias | Model | Size | mAP | Description |
|-------|-------|------|-----|-------------|
| `default` | RT-DETR R18 | ~80MB | 46.5 | Best balance of speed and accuracy (NMS-free) |
| `fast` | EfficientDet-D0 | ~15MB | 33.8 | Fastest inference, lightweight |
| `quality` | RT-DETR R50 | ~170MB | 53.1 | Higher accuracy, moderate speed |
| `large` | RT-DETR R101 | ~300MB | 54.3 | Highest accuracy |

All models use Apache-2.0 license and support 80 COCO classes.

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
