# LMSupply.Segmenter

A lightweight, zero-configuration image segmentation library for .NET with automatic GPU acceleration.

## Installation

```bash
dotnet add package LMSupply.Segmenter
```

Do not add ONNX Runtime packages (`Microsoft.ML.OnnxRuntime*`): LMSupply provisions the runtime itself, and a
second copy conflicts with it. `ExecutionProvider.Auto` uses CUDA when the CUDA 12 runtime and cuDNN 9 are
installed on the machine, CoreML on macOS, and the CPU otherwise. On Windows with an AMD or Intel GPU, ONNX
sessions run on the CPU (DirectML was removed in 0.67.0). See
[GPU acceleration](https://github.com/iyulab/lm-supply#gpu-acceleration).

## Basic Usage

```csharp
using LMSupply.Segmenter;

// Load the default model
await using var segmenter = await LocalSegmenter.LoadAsync("default");

// Perform semantic segmentation
var result = await segmenter.SegmentAsync("photo.jpg");

Console.WriteLine($"Image size: {result.Width}x{result.Height}");
Console.WriteLine($"Unique classes found: {result.UniqueClassCount}");

// Get class at a specific pixel
int classId = result.GetClassAt(100, 100);
Console.WriteLine($"Class at (100,100): {segmenter.ClassLabels[classId]}");
```

## Available Models

| Alias | Model | Size | mIoU | Description |
|-------|-------|------|------|-------------|
| `default` | SegFormer-B0 | ~15MB | 38.0 | Lightweight, fast inference |
| `fast` | SegFormer-B1 | ~55MB | 42.2 | Balanced speed and accuracy |
| `quality` | SegFormer-B2 | ~110MB | 46.5 | Higher accuracy |
| `large` | SegFormer-B5 | ~340MB | 51.0 | Highest accuracy |
| `interactive` | MobileSAM | ~45MB | - | Point/box prompt segmentation (`LoadInteractiveAsync`) |

All semantic segmentation models are trained on ADE20K (150 classes) with MIT license.

You can also use any HuggingFace segmentation model by its full ID:

```csharp
// Use any ONNX segmentation model from HuggingFace
var segmenter = await LocalSegmenter.LoadAsync("nvidia/segformer-b0-finetuned-ade-512-512");
```

## Advanced Usage

### Custom Options

```csharp
var options = new SegmenterOptions
{
    ResizeToOriginal = true,               // Resize output to match input dimensions
    Provider = ExecutionProvider.Cuda,     // Force specific GPU provider
    CacheDirectory = "/custom/cache"       // Custom model cache directory
};

var segmenter = await LocalSegmenter.LoadAsync("quality", options);
```

### Working with Segmentation Results

```csharp
var result = await segmenter.SegmentAsync("room.jpg");

// Get class coverage statistics
var coverage = result.GetClassCoveragePercentages();
foreach (var (classId, percentage) in coverage.OrderByDescending(x => x.Value).Take(5))
{
    Console.WriteLine($"{segmenter.ClassLabels[classId]}: {percentage:F1}%");
}
// Output:
// wall: 35.2%
// floor: 28.7%
// ceiling: 15.3%
// window: 8.1%
// furniture: 5.4%

// Get pixel counts per class
var pixelCounts = result.GetClassPixelCounts();

// Get binary mask for a specific class
int wallClassId = 0; // wall in ADE20K
bool[] wallMask = result.GetClassMask(wallClassId);
```

### Batch Processing

```csharp
var images = new[] { "image1.jpg", "image2.jpg", "image3.jpg" };
var results = await segmenter.SegmentBatchAsync(images);

for (int i = 0; i < images.Length; i++)
{
    Console.WriteLine($"{images[i]}: {results[i].UniqueClassCount} classes detected");
}
```

### Using Streams and Byte Arrays

```csharp
// From stream
using var stream = File.OpenRead("image.png");
var result = await segmenter.SegmentAsync(stream);

// From byte array (useful for API scenarios)
byte[] imageBytes = await httpClient.GetByteArrayAsync(imageUrl);
var result = await segmenter.SegmentAsync(imageBytes);
```

### Interactive Segmentation with MobileSAM

MobileSAM segments what a point or a box points at (0.88.0+; before, the alias pointed at a repository that did not exist
and no public method returned an interactive segmenter):

```csharp
await using var sam = await LocalSegmenter.LoadInteractiveAsync();          // "interactive" (MobileSAM, MIT)

// One session encodes the image once; each prompt then runs only the small decoder.
await using var session = await sam.CreateSessionAsync("photo.jpg");
var result = await session.SegmentAsync([new PointPrompt(520, 250, PointLabel.Foreground)]);
bool[,] mask = result.BestMask.Mask;                                          // [y, x], original image size

var boxed = await session.SegmentAsync(new BoxPrompt(400, 130, 240, 240));   // x, y, width, height
```

`LocalSegmenter.LoadAsync("interactive")` throws: the semantic-segmentation entry point cannot run a prompt-based model.

### Accessing Class Labels

```csharp
// Get all ADE20K class labels
var labels = LocalSegmenter.Ade20kClassLabels;
Console.WriteLine(string.Join(", ", labels.Take(5)));
// Output: wall, building, sky, floor, tree

// Or from the model instance
var modelLabels = segmenter.ClassLabels;
```

## GPU Acceleration

GPU acceleration is automatic when available. Priority order:
1. CUDA (NVIDIA GPUs)
2. CoreML (macOS)
3. CPU (fallback)

AMD / Intel GPUs on Windows have no ONNX provider on ONNX Runtime 1.25+ (DirectML was removed in 0.67.0); this module runs on CPU there.

Force a specific provider:

```csharp
var options = new SegmenterOptions
{
    Provider = ExecutionProvider.Cuda
};
```

## Model Caching

Models are cached following HuggingFace Hub conventions:
- Default: `~/.cache/huggingface/hub`
- Override via: `HF_HUB_CACHE`, `HF_HOME`, or `XDG_CACHE_HOME` environment variables
- Or set `SegmenterOptions.CacheDirectory`

## ADE20K Class Reference

The semantic segmentation models detect 150 ADE20K classes including:

| ID | Class | ID | Class | ID | Class |
|----|-------|----|----|----|----|
| 0 | wall | 10 | grass | 20 | sofa |
| 1 | building | 11 | sidewalk | 21 | shelf |
| 2 | sky | 12 | person | 22 | house |
| 3 | floor | 13 | earth | 23 | sea |
| 4 | tree | 14 | door | 24 | mirror |
| 5 | ceiling | 15 | table | 25 | rug |
| 6 | road | 16 | mountain | 26 | field |
| 7 | bed | 17 | plant | 27 | armchair |
| 8 | windowpane | 18 | curtain | 28 | seat |
| 9 | cabinet | 19 | chair | 29 | fence |

See [ADE20K dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/) for the complete list.
