# LMSupply.Vision.Core

Core vision processing infrastructure for LMSupply packages.

## Purpose

Provides shared image processing capabilities for vision-based AI packages:
- **LMSupply.Captioner** - Image captioning
- **LMSupply.Ocr** - Optical character recognition

## Key Components

- `IImageLoader` - Abstraction for loading images from various sources
- `IImagePreprocessor` - Model-specific image preprocessing pipeline
- `PreprocessProfile` - Configuration for image preprocessing parameters
- `TensorUtils` - Utilities for converting images to ONNX tensors

## Dependencies

- `SixLabors.ImageSharp` - Cross-platform image processing (no native dependencies)
- `LMSupply.Core` - Shared infrastructure (caching, downloading, ONNX utilities)

## Usage

```csharp
using LMSupply.Vision;

// Load and preprocess an image for a specific model
var preprocessor = new ImagePreprocessor();
var tensor = preprocessor.Preprocess("image.jpg", PreprocessProfiles.ImageNet);
```
