# LMSupply.Ocr

A lightweight, zero-configuration OCR library for .NET with automatic GPU acceleration. Features a 2-stage detection + recognition pipeline using PaddleOCR ONNX models.

## Installation

```bash
dotnet add package LMSupply.Ocr
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
using LMSupply.Ocr;

// Load default OCR pipeline (English)
await using var ocr = await LocalOcr.LoadAsync();

// Recognize text in an image
var result = await ocr.RecognizeAsync("document.png");

// Get all text
Console.WriteLine(result.FullText);

// Access individual text regions
foreach (var region in result.Regions)
{
    Console.WriteLine($"[{region.Confidence:P0}] {region.Text}");
    Console.WriteLine($"  Location: {region.BoundingBox}");
}
```

## Language-Specific OCR

```csharp
// Load OCR for Korean text
await using var ocr = await LocalOcr.LoadForLanguageAsync("ko");

// Or specify the recognition model explicitly
await using var ocr = await LocalOcr.LoadAsync(
    detectionModel: "default",
    recognitionModel: "crnn-korean-v3");
```

## Available Models

### Detection Models

| Alias | Model | Size | Description |
|-------|-------|------|-------------|
| `default` | DBNet v3 | ~2.3MB | Text detection for all languages |
| `dbnet-v3` | DBNet v3 | ~2.3MB | Same as default |

### Recognition Models

| Alias | Languages | Size |
|-------|-----------|------|
| `default` | English | ~7MB |
| `crnn-en-v3` | English | ~7MB |
| `crnn-korean-v3` | Korean | ~11MB |
| `crnn-chinese-v3` | Chinese (Simplified/Traditional) | ~13MB |
| `crnn-japan-v3` | Japanese | ~10MB |
| `crnn-latin-v3` | Spanish, French, German, Italian, Portuguese, etc. | ~7MB |
| `crnn-arabic-v3` | Arabic | ~7MB |
| `crnn-cyrillic-v3` | Russian, Ukrainian, Bulgarian, etc. | ~7MB |
| `crnn-devanagari-v3` | Hindi, Marathi, Nepali, Sanskrit | ~7MB |

## Configuration Options

```csharp
var options = new OcrOptions
{
    LanguageHint = "en",           // Language hint for auto model selection
    DetectionThreshold = 0.5f,      // Minimum detection confidence
    RecognitionThreshold = 0.5f,    // Minimum recognition confidence
    BinarizationThreshold = 0.3f,   // DBNet binarization threshold
    UnclipRatio = 1.5f,             // Polygon expansion ratio
    UsePolygon = true,              // Use polygon coordinates
    Provider = ExecutionProvider.Auto,  // GPU acceleration
    CacheDirectory = null           // Custom cache directory
};

await using var ocr = await LocalOcr.LoadAsync(options: options);
```

## Advanced Usage

### Detection Only

```csharp
// Get text regions without recognition
var regions = await ocr.DetectAsync("document.png");

foreach (var region in regions)
{
    Console.WriteLine($"Found text at: {region.BoundingBox}");
    Console.WriteLine($"Confidence: {region.Confidence:P0}");
}
```

### Layout-Aware Text Extraction

```csharp
var result = await ocr.RecognizeAsync("document.png");

// Get text with layout preserved (same-line regions joined with spaces)
var layoutText = result.GetTextWithLayout(lineTolerancePixels: 10);
```

### Using Streams and Byte Arrays

```csharp
// From stream
using var stream = File.OpenRead("document.png");
var result = await ocr.RecognizeAsync(stream);

// From byte array
byte[] imageData = await httpClient.GetByteArrayAsync(imageUrl);
var result = await ocr.RecognizeAsync(imageData);
```

### Polygon Coordinates

```csharp
var result = await ocr.RecognizeAsync("document.png");

foreach (var region in result.Regions)
{
    if (region.Polygon != null)
    {
        Console.WriteLine("Polygon points:");
        foreach (var point in region.Polygon)
        {
            Console.WriteLine($"  ({point.X}, {point.Y})");
        }
    }
}
```

## GPU Acceleration

GPU acceleration is automatic when available. Priority order:
1. CUDA (NVIDIA GPUs)
2. DirectML (Windows - AMD, Intel, NVIDIA)
3. CoreML (macOS)
4. CPU (fallback)

## Model Architecture

LMSupply.Ocr uses a 2-stage pipeline:

1. **Detection (DBNet)**: Differentiable Binarization Network detects text regions
2. **Recognition (CRNN)**: Convolutional Recurrent Neural Network with CTC decoding

Models are from the PaddleOCR project, converted to ONNX format:
- Repository: `monkt/paddleocr-onnx` on HuggingFace

## Model Caching

Models are cached following HuggingFace Hub conventions:
- Default: `~/.cache/huggingface/hub`
- Override via: `HF_HUB_CACHE`, `HF_HOME`, or `XDG_CACHE_HOME` environment variables
- Or set `OcrOptions.CacheDirectory`
