# LMSupply.Captioner

A lightweight, zero-configuration image captioning library for .NET with automatic GPU acceleration.

## Installation

```bash
dotnet add package LMSupply.Captioner
```

Do not add ONNX Runtime packages (`Microsoft.ML.OnnxRuntime*`): LMSupply provisions the runtime itself, and a
second copy conflicts with it. `ExecutionProvider.Auto` uses CUDA when the CUDA 12 runtime and cuDNN 9 are
installed on the machine, CoreML on macOS, and the CPU otherwise. On Windows with an AMD or Intel GPU, ONNX
sessions run on the CPU (DirectML was removed in 0.67.0). See
[GPU acceleration](https://github.com/iyulab/lm-supply#gpu-acceleration).

## Basic Usage

```csharp
using LMSupply.Captioner;

// Load the default model
await using var captioner = await LocalCaptioner.LoadAsync("default");

// Generate a caption from file
var result = await captioner.CaptionAsync("photo.jpg");
Console.WriteLine(result.Caption);
// Output: "A cat sitting on a windowsill looking outside"

// Generate a caption from stream
using var stream = File.OpenRead("image.png");
var result = await captioner.CaptionAsync(stream);
```

## Available Models

| Alias | Model | Download | Description |
|-------|-------|----------|-------------|
| `default` (also `quality`, `auto`) | Florence-2 base (MIT) | ~275 MB int8 · ~545 MB fp16 · ~1.1 GB fp32 | Names the main subjects of everyday photos; brief, detailed or paragraph captions |
| `fast` | ViT-GPT2 (Apache-2.0) | ~960 MB | One-sentence captions; tends to name a familiar scene rather than the subject |

`default` was ViT-GPT2 before 0.104.0; load `fast` (or `Xenova/vit-gpt2-image-captioning`) to keep it.

Florence-2 is published in quantized variants. A load takes the variant this machine's hardware tier picks (int8 on most
machines), or the one named by `CaptionerOptions.QuantizationHint` or a qualifier (`"quality:fp16"`).
`LocalCaptioner.GetDownloadSizeBytesAsync(alias, options)` answers what that load will download — for a consent screen —
without downloading anything.

### Level of detail

```csharp
using LMSupply.Captioner;

await using var captioner = await LocalCaptioner.LoadAsync("quality",
    new CaptionerOptions { Detail = CaptionDetail.Paragraph, MaxLength = 150 });
var result = await captioner.CaptionAsync("photo.jpg");
```

Decoding is greedy by default. `NumBeams = 3` (for example) runs a beam search and returns the runners-up in
`CaptionResult.AlternativeCaptions`, at one decoder pass per beam per token; `Temperature = 0.7f` samples instead.
The two are exclusive.

`Detail` is `Brief` (default, one sentence), `Detailed` or `Paragraph`; a paragraph needs a larger `MaxLength` than the
default 50. Only `quality` reads it — `default` captions at one level and refuses any other value at load
(`NotSupportedException`). The reverse holds for `Prompt`: `default` continues a caption from it, `quality` refuses it
because its prompt selects a task.

You can also use any HuggingFace vision-language model by its full ID:

```csharp
// Use any ONNX captioning model from HuggingFace
var captioner = await LocalCaptioner.LoadAsync("Xenova/vit-gpt2-image-captioning");
```

## Advanced Usage

### Custom Options

```csharp
var options = new CaptionerOptions
{
    MaxLength = 50,                        // Maximum caption length
    Provider = ExecutionProvider.Cuda,     // Force specific GPU provider
    CacheDirectory = "/custom/cache",      // Custom model cache directory
    DisableAutoDownload = false            // true: load only from the cache, throw if a file is missing
};

var captioner = await LocalCaptioner.LoadAsync("default", options);
var result = await captioner.CaptionAsync("image.jpg");

Console.WriteLine($"Caption: {result.Caption}");
Console.WriteLine($"Confidence: {result.Confidence:P1}");
```

### Conditional captioning

`Prompt` sets the words the caption starts with; the model continues from them:

```csharp
await using var captioner = await LocalCaptioner.LoadAsync("default", new CaptionerOptions { Prompt = "a painting of" });
var result = await captioner.CaptionAsync("image.jpg");   // "a painting of a colorful wall with ..."
Console.WriteLine($"Processing time: {result.ProcessingTimeMs}ms");
```

### Batch Processing

```csharp
var images = new[] { "image1.jpg", "image2.jpg", "image3.jpg" };

foreach (var image in images)
{
    var result = await captioner.CaptionAsync(image);
    Console.WriteLine($"{image}: {result.Caption}");
}
```

### Using Byte Arrays

```csharp
// Caption from byte array (useful for API scenarios)
byte[] imageBytes = await httpClient.GetByteArrayAsync(imageUrl);
var result = await captioner.CaptionAsync(imageBytes);
```

## GPU Acceleration

GPU acceleration is automatic when available. Priority order:
1. CUDA (NVIDIA GPUs)
2. CoreML (macOS)
3. CPU (fallback)

AMD / Intel GPUs on Windows have no ONNX provider on ONNX Runtime 1.25+ (DirectML was removed in 0.67.0); this module runs on CPU there.

Force a specific provider:

```csharp
var options = new CaptionerOptions
{
    Provider = ExecutionProvider.Cuda
};
```

## Model Caching

Models are cached following HuggingFace Hub conventions:
- Default: `~/.cache/huggingface/hub`
- Override via: `HF_HUB_CACHE`, `HF_HOME`, or `XDG_CACHE_HOME` environment variables
- Or set `CaptionerOptions.CacheDirectory`

Set `CaptionerOptions.DisableAutoDownload = true` to load from the cache only: a model file that is not
there throws `ModelNotFoundException`, no network request is made, and nothing is written to the cache.
Use it behind a download-consent boundary — the consenting call loads with downloads on, every later
call with the flag set — so a model that is not installed fails instead of downloading.
