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

### Korean OCR

```csharp
// Load OCR for Korean text
await using var ocr = await LocalOcr.LoadForLanguageAsync("ko");

var result = await ocr.RecognizeAsync("korean_document.png");
Console.WriteLine(result.FullText);

// Korean + English mixed documents work with the Korean model
// The CRNN model recognizes both Hangul and Latin characters
```

### Other Languages

```csharp
// Japanese (read by the Chinese/Japanese recognizer)
await using var ocr = await LocalOcr.LoadForLanguageAsync("ja");

// Chinese
await using var ocr = await LocalOcr.LoadForLanguageAsync("zh");

// Or specify the recognition model explicitly
await using var ocr = await LocalOcr.LoadAsync(
    detectionModel: "default",
    recognitionModel: "crnn-chinese-v3");
```

### Which languages can be recognized

Recognition is per script and the caller picks the language — the models have no script or language
identification output, so there is no detection step. Ask the library which codes resolve to a model:

```csharp
IEnumerable<string> codes = LocalOcr.GetSupportedLanguages();   // "en", "ko", "ja", "tr", ...
```

Region subtags are ignored (`"tr-TR"` resolves as `"tr"`). A code with no model falls back to the English
recognizer and raises a trace warning naming the code — the English model cannot read another script, so
its output for such text would otherwise look like a result.

## Available Models

### Detection Models

| Alias | Model | Size | Description |
|-------|-------|------|-------------|
| `default` | DBNet v3 | ~2.3MB | Text detection for all languages |
| `dbnet-v3` | DBNet v3 | ~2.3MB | Same as default |

### Recognition Models

| Alias | Languages |
|-------|-----------|
| `default` | English |
| `crnn-en-v3` | English |
| `crnn-korean-v3` | Korean |
| `crnn-chinese-v3` | Chinese (Simplified/Traditional), Japanese |
| `crnn-latin-v3` | Spanish, French, German, Italian, Portuguese, Dutch, Polish, Czech, Slovak, Turkish, Hungarian, Nordic, Baltic, Indonesian, Malay, Filipino, and more |
| `crnn-arabic-v3` | Arabic, Urdu, Persian |
| `crnn-cyrillic-v3` | Russian, Ukrainian, Bulgarian, Belarusian, Serbian, Macedonian |
| `crnn-devanagari-v3` | Hindi, Marathi, Nepali, Sanskrit |
| `crnn-greek-v3` | Greek |
| `crnn-thai-v3` | Thai |
| `crnn-tamil-v3` | Tamil |
| `crnn-telugu-v3` | Telugu |

You can also use any HuggingFace PaddleOCR-compatible model by its full repository ID:

```csharp
// Use any PaddleOCR-style ONNX model from HuggingFace
// The model must have detection model (det.onnx) and recognition model (rec.onnx + dict.txt)

// Using a custom PaddleOCR ONNX repository; LanguageHint picks a languages/<name>/ subfolder
await using var ocr = await LocalOcr.LoadAsync(
    detectionModel: "your-org/custom-paddleocr",
    recognitionModel: "your-org/custom-paddleocr"
);
```

### HuggingFace Model Requirements

For custom HuggingFace OCR repositories, the following file structure is expected:

**Detection Model** (one of):
- `det.onnx`, `detection.onnx`, `text_detection.onnx`, or `detector.onnx`
- Can be in root or subfolders: `detection/`, `detection/v5/`, `onnx/`

**Recognition Model** (both required):
- Model file: `rec.onnx`, `recognition.onnx`, `text_recognition.onnx`, or `recognizer.onnx`
- Dictionary file: `dict.txt`, `dictionary.txt`, `keys.txt`, or `vocab.txt`
- Can be in language-specific subfolders: `languages/english/`, `languages/korean/`, etc.

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
