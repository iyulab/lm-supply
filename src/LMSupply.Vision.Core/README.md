# LMSupply.Vision.Core

Core vision processing infrastructure for LMSupply packages.

## Purpose

Provides shared image processing capabilities for vision-based AI packages:
- **LMSupply.Captioner** - Image captioning
- **LMSupply.Ocr** - Optical character recognition

## Key Components

- `RgbImage` - 8-bit RGB image in memory, with bicubic `Resize`, `Crop` and `Letterbox`
- `IImageLoader` / `ImageLoader` - Decodes JPEG, PNG, WebP, GIF (first frame), BMP and ICO; `ImageLoader.EncodePng` writes PNG
- `IImagePreprocessor` - Model-specific image preprocessing pipeline
- `PreprocessProfile` - Configuration for image preprocessing parameters
- `TensorUtils` - Utilities for converting images to ONNX tensors

## Dependencies

- `SkiaSharp` - Image decoding and PNG encoding (MIT). Native libraries for Windows, macOS and Linux come with the package; nothing to add per platform
- `LMSupply.Core` - Shared infrastructure (caching, downloading, ONNX utilities)

## Usage

```csharp
using LMSupply.Vision;

// Load and preprocess an image for a specific model
var preprocessor = new ImagePreprocessor();
float[] tensor = await preprocessor.PreprocessAsync("image.jpg", PreprocessProfile.ImageNet);
```
