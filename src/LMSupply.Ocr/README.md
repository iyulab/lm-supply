# LMSupply.Ocr

A simple .NET library for local OCR (Optical Character Recognition) with automatic model downloading from HuggingFace. Features a 2-stage detection + recognition pipeline using PaddleOCR ONNX models.

## Features

- **2-Stage Pipeline**: Text detection (DBNet) followed by text recognition (CRNN with CTC decoding)
- **Multi-language Support**: 40+ languages including English, Korean, Chinese, Japanese, Arabic, and more
- **Automatic Model Download**: Models are downloaded on-demand from HuggingFace (~10MB default)
- **GPU Acceleration**: Supports CUDA, DirectML, and CoreML
- **Pure C# Implementation**: No Python dependencies or external processes
- **Polygon Support**: Precise text region boundaries for rotated or curved text

## Quick Start

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

## Supported Languages

| Model | Languages |
|-------|-----------|
| `crnn-en-v3` | English |
| `crnn-korean-v3` | Korean |
| `crnn-chinese-v3` | Chinese (Simplified/Traditional) |
| `crnn-japan-v3` | Japanese |
| `crnn-latin-v3` | Spanish, French, German, Italian, Portuguese, etc. |
| `crnn-arabic-v3` | Arabic |
| `crnn-cyrillic-v3` | Russian, Ukrainian, Bulgarian, etc. |
| `crnn-devanagari-v3` | Hindi, Marathi, Nepali, Sanskrit |

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

## Detection Only

```csharp
// Get text regions without recognition
var regions = await ocr.DetectAsync("document.png");

foreach (var region in regions)
{
    Console.WriteLine($"Found text at: {region.BoundingBox}");
}
```

## Layout-Aware Text Extraction

```csharp
var result = await ocr.RecognizeAsync("document.png");

// Get text with layout preserved (same-line regions joined with spaces)
var layoutText = result.GetTextWithLayout(lineTolerancePixels: 10);
```

## Model Information

The library uses PaddleOCR ONNX models from HuggingFace:
- Repository: `monkt/paddleocr-onnx`
- Detection model: DBNet (~2.3MB)
- Recognition models: CRNN (~7-13MB depending on language)

Models are cached locally after first download.
