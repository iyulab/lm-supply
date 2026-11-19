# LMSupply.Ocr

A simple .NET library for local OCR (Optical Character Recognition) with automatic model downloading from HuggingFace. Features a 2-stage detection + recognition pipeline using PaddleOCR ONNX models.

## Features

- **2-Stage Pipeline**: Text detection (DBNet) followed by text recognition (CRNN with CTC decoding)
- **Multi-language Support**: 11 script recognizers covering 50+ language codes, including English, Korean, Chinese, Japanese, Arabic, Cyrillic and most Latin-script languages
- **Automatic Model Download**: Models are downloaded on-demand from HuggingFace (~10MB default)
- **GPU Acceleration**: Supports CUDA and CoreML
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
await using var korean = await LocalOcr.LoadForLanguageAsync("ko");

// Or specify the recognition model explicitly
await using var explicitModels = await LocalOcr.LoadAsync(
    detectionModel: "default",
    recognitionModel: "crnn-korean-v5");
```

## Supported Languages

Recognition is per script: each model reads one script's alphabet, and the caller picks the language.
There is no script or language detection step — the models have no identification output to detect with.

| Model | Languages |
|-------|-----------|
| `crnn-en-v5` | English |
| `crnn-korean-v5` | Korean |
| `crnn-chinese-v5` | Chinese (Simplified/Traditional), Japanese |
| `crnn-latin-v5` | Spanish, French, German, Italian, Portuguese, Dutch, Polish, Czech, Slovak, Turkish, Hungarian, Nordic, Baltic, Indonesian, Malay, Filipino, Latvian, Welsh, Afrikaans, and more |
| `crnn-arabic-v5` | Arabic, Urdu, Persian |
| `crnn-cyrillic-v5` | Russian, Ukrainian, Bulgarian, Belarusian, Serbian, Macedonian |
| `crnn-devanagari-v5` | Hindi, Marathi, Nepali, Sanskrit |
| `crnn-greek-v5` | Greek |
| `crnn-thai-v5` | Thai |
| `crnn-tamil-v5` | Tamil |
| `crnn-telugu-v5` | Telugu |

The suffix names the PP-OCR generation of the model behind the alias.

Ask the library which codes it can resolve, rather than keeping your own list:

```csharp
IEnumerable<string> codes = LocalOcr.GetSupportedLanguages();   // "en", "ko", "ja", "tr", ...
```

A code is read as a BCP 47 tag or a POSIX locale name: `-` and `_` both separate subtags, case is ignored, and an
encoding or modifier is dropped except `@latin` and `@cyrillic`, which name the script. The script comes before the
region, so `"sr-Latn"` is the Latin recognizer's and `"sr"` the Cyrillic one's, and a region the library has no model
for is ignored (`"tr-TR"` and `"ko_KR.UTF-8"` resolve as `"tr"` and `"ko"`). `cmn` and `yue` resolve as `zh`.

A code that names a script the language's recognizer does not read (`"bs-Cyrl"`, `"mk-Latn"`) is refused too, not read with the recognizer of the bare language.

A code with no model is refused: the English recognizer cannot read another script, and it would still return text,
so a call that names such a language throws `UnsupportedOcrLanguageException` — an `ArgumentException` that carries the
code and the supported codes — and `LocalOcr.DeleteLanguageModel` returns `false`. Pass `"en"` to read with the English
recognizer on purpose, or `OcrRecognitionModelRegistry.Default.TryResolveForLanguage` to ask without an exception.

## Configuration Options

```csharp
using LMSupply;   // ExecutionProvider

var options = new OcrOptions
{
    LanguageHint = "en",           // Language hint for auto model selection
    DetectionThreshold = 0.5f,      // Minimum detection confidence
    RecognitionThreshold = 0.5f,    // Regions below this recognition confidence are dropped (0 keeps all)
    BinarizationThreshold = 0.3f,   // DBNet binarization threshold
    UnclipRatio = 1.5f,             // Polygon expansion ratio
    UsePolygon = true,              // Use polygon coordinates
    Provider = ExecutionProvider.Auto,  // GPU acceleration
    CacheDirectory = null,          // Custom cache directory
    DisableAutoDownload = false     // true: load only from the cache, throw if a file is missing
};

await using var ocr = await LocalOcr.LoadAsync(options: options);
```

## Checking the Cache Before Loading

Loading a language downloads whatever it needs that is not cached. To ask first — before a download
the user has not agreed to — check the language's cache status. It makes no network request:

```csharp
var status = LocalOcr.GetCacheStatusForLanguage("ko");
if (!status.IsCached)
{
    foreach (var file in status.Missing)
        Console.WriteLine($"Would download {file.RepoId}/{file.Subfolder}/{file.FileName}");
}
```

The files checked are the ones `LoadForLanguageAsync` loads: the detection model plus the language's
recognizer and dictionary. Languages of one script share a recognizer, so having English cached says
nothing about Korean. `DisableAutoDownload = true` makes a load throw instead of downloading.

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
